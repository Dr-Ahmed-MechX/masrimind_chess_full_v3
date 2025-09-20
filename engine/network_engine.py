# engine/network_engine.py
import os, torch
import chess

try:
    from engine.policy_value_net import PVNetPolicyValue
    from training.encode import fen_to_tensor as encode_fen_to_tensor
    _HAS_MODEL = True
except Exception:
    _HAS_MODEL = False

def network_available():
    return _HAS_MODEL

class NetworkEngine:
    def __init__(self, model_path="training/pvnet_pv_rl.pt", device=None):
        self.ready = False
        if not _HAS_MODEL:
            return
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = PVNetPolicyValue().to(self.device).eval()
        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.ready = True

    def policy_over_legal(self, board: chess.Board, legal_moves):
        # يُرجع احتمالات policy بنفس ترتيب legal_moves
        if not self.ready:
            return [1.0/len(legal_moves)]*len(legal_moves)
        x = encode_fen_to_tensor(board.fen()).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device.split(':')[0]):
            out = self.model(x)
            logits_all = out["policy_logits"]  # نتوقع [B, M]
            # في حال موديلك بيطلع حجم ثابت > len(legal_moves)، قص للـ len(legal_moves)
            logits = logits_all[0, :len(legal_moves)]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().tolist()
            return probs

    def best_move(self, board: chess.Board):
        legal = list(board.legal_moves)
        if not legal:
            return None
        probs = self.policy_over_legal(board, legal)
        idx = max(range(len(legal)), key=lambda i: probs[i])
        return legal[idx]
