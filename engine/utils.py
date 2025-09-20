import glob, os, torch
def latest_weight(globpat):
    files=glob.glob(globpat); 
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]
def mv_index(move):
    # (from,to) into [0..4095]
    return move.from_square * 64 + move.to_square
