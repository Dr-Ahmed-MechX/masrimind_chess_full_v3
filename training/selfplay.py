def main():
    import argparse, os, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int)
    ap.add_argument("--mcts-simulations", type=int)
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("Stub: self-playing games...")
    time.sleep(1)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "episodes.jsonl"), "w") as f:
        f.write(json.dumps({"game": 1, "moves": ["e2e4", "e7e5"]}) + "\n")
    print("Saved self-play data to", args.out)

if __name__ == "__main__":
    main()
