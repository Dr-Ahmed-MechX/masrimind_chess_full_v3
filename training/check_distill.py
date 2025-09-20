import json

path = "training/distill.jsonl"
bad, total = 0, 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        line = line.strip()
        if not line:
            bad += 1
            continue
        try:
            js = json.loads(line)
            assert "fen" in js and "eval_cp" in js
        except:
            bad += 1

print(f"✅ Total: {total}, ❌ Bad: {bad}, ✅ Good: {total - bad}")
