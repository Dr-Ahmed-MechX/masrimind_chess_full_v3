# MasriMind — Training Pro (Real FEN encoding + CNN/Transformer + Stockfish Distillation)

This pack adds:
- Real FEN → tensor encoder (12 piece planes + stm/castling/ep/ply clocks).
- Two model options: **PVNetCNN** (fast) and **TinyTransformer** (small transformer).
- Dataset builder that queries **Stockfish** to produce evaluation targets (centipawns).
- Training script for **value distillation** (regress normalized value in [-1, 1]).

## Quick Start
```bash
# 1) Generate dataset with Stockfish
python training/make_dataset_distill.py --games 800 --max-plies 80 --depth 14

# 2) Train (CNN by default)
python training/train_distill_real.py --model cnn --epochs 6 --batch-size 512

# or Transformer
python training/train_distill_real.py --model transformer --epochs 8 --batch-size 256

# 3) The script will save weights as: training/pvnet_epoch{N}.pt
#    Your app auto-loads the latest matching weights.
```
