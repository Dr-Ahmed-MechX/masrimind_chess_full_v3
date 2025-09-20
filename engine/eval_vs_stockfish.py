# training/eval_vs_stockfish.py
# Run matches: (HybridChooser (PVNet+MCTS) vs Stockfish) and save CSV + PGN

import os, sys, csv, argparse, time, yaml
import chess, chess.pgn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from engine.chooser import HybridChooser
from engine.uci_stockfish import StockfishUCI

CFG = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8"))

def play_game(ai_white: bool, rollouts=None, c_puct=None, sf_move_ms=None, verbose=False):
    board = chess.Board()
    if rollouts is not None or c_puct is not None:
        # مؤقتًا نغيّر إعدادات الـMCTS في الرن دي فقط
        CFG['engine']['mcts_rollouts'] = int(rollouts if rollouts is not None else CFG['engine']['mcts_rollouts'])
        CFG['engine']['c_puct'] = float(c_puct if c_puct is not None else CFG['engine']['c_puct'])

    chooser = HybridChooser(board)
    sf = StockfishUCI()
    if sf_move_ms is not None:
        # إعادة بناء الليمت بزمن مختلف
        sf.limit = chess.engine.Limit(time=max(0.05, float(sf_move_ms)/1000.0))

    game = chess.pgn.Game()
    game.headers["Event"] = "AI vs Stockfish"
    game.headers["White"] = "AI" if ai_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if ai_white else "AI"
    node = game

    moves = 0
    t0 = time.time()
    while not board.is_game_over() and moves < 512:
        if (board.turn == chess.WHITE and ai_white) or (board.turn == chess.BLACK and not ai_white):
            mv = chooser.best_move()
        else:
            mv = sf.best_move(board)
        board.push(mv)
        node = node.add_variation(mv)
        moves += 1

    dur = time.time() - t0
    res = board.result()  # "1-0","0-1","1/2-1/2"
    chooser.close()
    sf.close()
    return res, game, dur, moves

def result_to_score(res, ai_white):
    # returns (ai_score, sf_score) with AI POV
    if res == "1-0":
        return (1.0, 0.0) if ai_white else (0.0, 1.0)
    if res == "0-1":
        return (0.0, 1.0) if ai_white else (1.0, 0.0)
    return (0.5, 0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=None, help="Override MCTS rollouts for this eval run")
    ap.add_argument("--c_puct", type=float, default=None, help="Override c_puct for this eval run")
    ap.add_argument("--sf_move_ms", type=int, default=None, help="Stockfish per-move time ms (override)")
    ap.add_argument("--csv", type=str, default="training/eval_results.csv")
    ap.add_argument("--pgn", type=str, default="training/eval_results.pgn")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.pgn) or ".", exist_ok=True)

    ai_points = 0.0
    sf_points = 0.0
    games = args.games

    with open(args.csv, "w", newline="", encoding="utf-8") as fcsv, open(args.pgn, "w", encoding="utf-8") as fpgn:
        wr = csv.writer(fcsv)
        wr.writerow(["game", "ai_color", "result", "ai_score", "sf_score", "moves", "seconds"])

        for i in range(1, games+1):
            ai_white = (i % 2 == 1)  # alternate colors
            res, game, secs, moves = play_game(ai_white, args.rollouts, args.c_puct, args.sf_move_ms)
            ai_s, sf_s = result_to_score(res, ai_white)
            ai_points += ai_s
            sf_points += sf_s

            wr.writerow([i, "white" if ai_white else "black", res, ai_s, sf_s, moves, round(secs,2)])
            fpgn.write(str(game) + "\n\n")
            print(f"Game {i:02d} | AI({ 'W' if ai_white else 'B' }) vs SF -> {res} | {moves} moves | {secs:.1f}s")

    print(f"\nAI total: {ai_points} / {games}  |  SF total: {sf_points} / {games}")
    score = ai_points / games
    # تقدير إلو تقريبي من النتائج: elo ~ 400 * log10(score/(1-score))
    import math
    if 0 < score < 1:
        elo = 400.0 * math.log10(score / (1.0 - score))
        print(f"Approx Elo vs SF (AI stronger if +): {elo:+.1f}")
    else:
        print("Elo undefined (perfect wins or losses).")

if __name__ == "__main__":
    main()
