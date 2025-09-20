import os
import torch
import yaml

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_model():
    cfg = load_config()
    model_path = (cfg.get("engine", {}) or {}).get("network_model") or (cfg.get("paths", {}) or {}).get("pvnet_model")
    device = cfg["model"].get("device", "cpu")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please place pvnet_pv_rl.pt there or update config.yaml -> model.policy_value_model_path"
        )
    # التحميل الفعلي بيكون في policy_value_net أو السيرفر. هنا بس نتحقق من وجوده.
    print(f"[ensure_model] Found model: {model_path} (device={device})")

if __name__ == "__main__":
    ensure_model()
