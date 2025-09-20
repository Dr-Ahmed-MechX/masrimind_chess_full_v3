# engine/mcts.py
# PUCT MCTS for PVNetPolicyValue — returns (best_uci, stats) where
# stats = {"visits": {uci:N}, "priors": {uci:P}, "q": {uci:Q}}

import math
import torch
import chess

from training.encode import fen_to_tensor
# الشبكة المتوقعة: forward(planes, scalars) -> (logits[4096], value[1])

# ---------- Helpers ----------
def mv_index(move: chess.Move) -> int:
    """Map (from,to) to [0..4095] index."""
    return move.from_square * 64 + move.to_square

def legal_mask(board: chess.Board) -> torch.Tensor:
    """Boolean mask (4096,) for legal moves."""
    mask = torch.zeros(64 * 64, dtype=torch.bool)
    for m in board.legal_moves:
        mask[mv_index(m)] = True
    return mask

# ---------- Node ----------
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior: float = 0.0):
        # نحتفظ بنسخة خفيفة من البورد (من غير stack لمزيد من السرعة)
        self.board = board.copy(stack=False)
        self.parent = parent
        self.prior = float(prior)  # P(s,a)
        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.N = 0                 # visit count
        self.W = 0.0               # total value
        self.Q = 0.0               # mean value
        self._expanded = False

    def is_leaf(self) -> bool:
        return not self._expanded

# ---------- MCTS ----------
class MCTS:
    def __init__(
        self,
        net=None,
        c_puct: float = 1.4,
        rollouts: int = 400,
        device: str = "cpu",
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.25,
        max_depth: int = 512,
    ):
        """
        net: شبكة Policy+Value (PVNetPolicyValue) أو None (يسقط على priors uniform وقيمة 0)
        """
        self.net = net.eval() if net is not None else None
        self.c_puct = float(c_puct)
        self.rollouts = int(rollouts)
        self.device = device
        self.root_dirichlet_alpha = float(root_dirichlet_alpha)
        self.root_dirichlet_eps = float(root_dirichlet_eps)
        self.max_depth = int(max_depth)

    # --------- Core steps ---------
    def ucb(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Q + U, حيث U = c_puct * P * sqrt(N_parent) / (1 + N_child)"""
        U = self.c_puct * child.prior * math.sqrt(parent.N + 1e-9) / (1.0 + child.N)
        return child.Q + U

    def select(self, node: MCTSNode) -> MCTSNode:
        """امشي لحد ورقة باستخدام UCB."""
        depth = 0
        while (not node.is_leaf()) and (not node.board.is_game_over()) and depth < self.max_depth:
            # اختَر الطفل بأعلى UCB
            best_m, best_ch, best_score = None, None, -1e18
            for m, ch in node.children.items():
                s = self.ucb(node, ch)
                if s > best_score:
                    best_score, best_m, best_ch = s, m, ch
            node = best_ch
            depth += 1
        return node

    @torch.no_grad()
    def _net_infer(self, board: chess.Board):
        """
        إرجع priors (dict move->P) و value:
        - لو self.net = None: priors uniform على القانوني والقيمة 0
        """
        legals = list(board.legal_moves)
        if not legals:
            return {}, 0.0

        if self.net is None:
            # uniform prior + zero value
            p = 1.0 / len(legals)
            priors = {m: p for m in legals}
            return priors, 0.0

        planes, scalars = fen_to_tensor(board.fen())
        planes = planes.unsqueeze(0).to(self.device).float()
        scalars = scalars.unsqueeze(0).to(self.device).float()

        logits, v = self.net(planes, scalars)  # (1,4096), (1,1)
        logits = logits.squeeze(0)

        mask = legal_mask(board).to(self.device)  # (4096,)
        logits = logits.masked_fill(~mask, -1e9)
        prob = torch.softmax(logits, dim=-1).detach().cpu()

        priors = {}
        for m in legals:
            priors[m] = float(prob[mv_index(m)])
        return priors, float(v.item())

    def expand(self, node: MCTSNode):
        """Expand node with priors from net (أو uniform)."""
        board = node.board
        if board.is_game_over():
            node._expanded = True
            return  # لا أطفال في وضع نهاية اللعبة

        priors, _ = self._net_infer(board)
        # أنشئ الأطفال
        for m in board.legal_moves:
            b2 = board.copy(stack=False)
            b2.push(m)
            node.children[m] = MCTSNode(b2, parent=node, prior=priors.get(m, 0.0))

        node._expanded = True

    def add_dirichlet_to_root(self, root: MCTSNode):
        """Dirichlet noise على الجذر لزيادة الاستكشاف."""
        if not root.children:
            return
        try:
            import numpy as np
        except Exception:
            return
        moves = list(root.children.keys())
        alphas = [self.root_dirichlet_alpha] * len(moves)
        noise = np.random.dirichlet(alphas)
        for i, m in enumerate(moves):
            child = root.children[m]
            child.prior = (1.0 - self.root_dirichlet_eps) * child.prior + self.root_dirichlet_eps * float(noise[i])

    def evaluate(self, node: MCTSNode) -> float:
        """
        قيمة الورقة من منظور اللاعب الذي عليه الدور في 'node'.
        - لو نهاية (checkmate على اللاعب الحالي): -1
        - لو تعادل: 0
        - لو لسه: استخدم الشبكة لإعطاء value (لكن التوسّع بيحصل في expand)
        """
        board = node.board
        if board.is_game_over():
            if board.is_checkmate():
                # اللاعب الحالي لا يملك نقلات => خاسر
                return -1.0
            return 0.0
        # لو مش نهاية، هنكتفي بـ 0 هنا لأننا بنعمل expand + net_infer لاحقًا باللفة
        return 0.0

    def backprop(self, node: MCTSNode, value: float):
        """إرجاع القيمة لأعلى الشجرة مع قلب الإشارة بالتناوب."""
        cur = node
        v = float(value)
        while cur is not None:
            cur.N += 1
            cur.W += v
            cur.Q = cur.W / cur.N
            v = -v  # flip POV كل مستوى
            cur = cur.parent

    def run(self, root: MCTSNode):
        """
        نفّذ rollouts وتحديث الشجرة.
        يرجّع: (best_uci, stats) حيث stats فيه visits/priors/q عند الجذر.
        """
        # لو الجذر ليس موسّعًا، وسّعه وخليك تضيف Dirichlet
        if root.is_leaf() and not root.board.is_game_over():
            self.expand(root)
            self.add_dirichlet_to_root(root)

        # رولات
        for _ in range(self.rollouts):
            # 1) SELECT
            leaf = self.select(root)

            # 2) EVAL + EXPAND
            if leaf.board.is_game_over():
                v = self.evaluate(leaf)
                # نهاية: لا توسّع
            else:
                # احصل على priors, value من الشبكة ووسّع العقدة
                priors, v = self._net_infer(leaf.board)
                # expand children with priors
                for m in leaf.board.legal_moves:
                    if m not in leaf.children:
                        b2 = leaf.board.copy(stack=False)
                        b2.push(m)
                        leaf.children[m] = MCTSNode(b2, parent=leaf, prior=priors.get(m, 0.0))
                leaf._expanded = True

            # 3) BACKUP
            self.backprop(leaf, v)

        # --------- إخراج الإحصاءات ---------
        if not root.children:
            return None, {"visits": {}, "priors": {}, "q": {}}

        visits = {m.uci(): ch.N for m, ch in root.children.items()}
        priors = {m.uci(): float(ch.prior) for m, ch in root.children.items()}
        qvals = {m.uci(): float(ch.Q) for m, ch in root.children.items()}

        best_uci = max(visits.items(), key=lambda kv: kv[1])[0]
        stats = {"visits": visits, "priors": priors, "q": qvals}
        return best_uci, stats
