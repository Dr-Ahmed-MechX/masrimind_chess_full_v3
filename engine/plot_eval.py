# training/plot_eval.py
# Read eval_results.csv and plot winrate, draw rate, elo estimate timeline

import argparse, csv, math
from collections import defaultdict
import matplotlib.pyplot as plt

def approx_elo(score):
    if score <= 0 or score >= 1: return None
    return 400.0 * math.log10(score/(1.0-score))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training/eval_results.csv")
    ap.add_argument("--out", default="training/eval_plot.png")
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)

    if not rows:
        print("No rows found.")
        return

    # cumulative stats
    xs, wrs, drs, lrs, elos = [], [], [], [], []
    ai_pts = 0.0
    for i, r in enumerate(rows, 1):
        res = r["result"]
        if res == "1/2-1/2":
            ai_pts += 0.5
        elif res == "1-0":
            ai_pts += 1.0 if r["ai_color"] == "white" else 0.0
        elif res == "0-1":
            ai_pts += 1.0 if r["ai_color"] == "black" else 0.0

        so_far = rows[:i]
        wins = 0; draws = 0; losses = 0
        for rr in so_far:
            res2 = rr["result"]
            if res2 == "1/2-1/2": draws += 1
            elif res2 == "1-0":
                wins += 1 if rr["ai_color"] == "white" else 0
                losses += 1 if rr["ai_color"] == "black" else 0
            elif res2 == "0-1":
                wins += 1 if rr["ai_color"] == "black" else 0
                losses += 1 if rr["ai_color"] == "white" else 0

        xs.append(i)
        wrs.append(wins/i)
        drs.append(draws/i)
        lrs.append(losses/i)
        sc = ai_pts / i
        elos.append(approx_elo(sc))

    # plot
    fig = plt.figure(figsize=(9,5))
    plt.plot(xs, wrs, label="Win rate")
    plt.plot(xs, drs, label="Draw rate")
    plt.plot(xs, lrs, label="Loss rate")
    plt.ylabel("Rate")
    plt.xlabel("Games")
    plt.ylim(0,1)
    plt.legend(loc="best")
    plt.twinx()
    plt.plot(xs, [e if e is not None else float('nan') for e in elos], label="Approx Elo vs SF")
    plt.ylabel("Approx Elo")
    plt.title("Eval vs Stockfish")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print("Saved plot:", args.out)

if __name__ == "__main__":
    main()
