import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str], cwd: Optional[Path]=None, env: Optional[Dict[str, str]]=None) -> int:
    log(f"Running: {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    log(f"Finished with code {proc.returncode}")
    return proc.returncode

def which(name: str) -> Optional[str]:
    return shutil.which(name)

def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False

def symlink_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def dump_yaml(obj: Dict[str, Any], path: Path):
    import yaml
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def timestamp_dir(base: Path, prefix: str) -> Path:
    d = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ensure_dir(d)
    return d
