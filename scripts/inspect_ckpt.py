"""
Inspect a PyTorch/PL checkpoint and print epoch/global_step and checkpoint keys.
Run: python scripts/inspect_ckpt.py <path/to/ckpt>
"""
import sys
import json

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Usage: python inspect_ckpt.py <checkpoint_path>")
    sys.exit(2)

try:
    import torch
except Exception as e:
    print("ERROR: torch not available:", e)
    sys.exit(3)

try:
    ckpt = torch.load(path, map_location="cpu")
except Exception as e:
    print("ERROR: failed to load checkpoint:", e)
    sys.exit(4)

out = {}
# Top-level keys and types
out["keys"] = {k: type(v).__name__ for k, v in ckpt.items()} if isinstance(ckpt, dict) else {"root": type(ckpt).__name__}

# Common PL fields
candidates = ["epoch", "global_step", "global_step0", "step", "state_dict", "trainer", "callbacks", "hyper_parameters", "pytorch-lightning_version"]
for k in candidates:
    if isinstance(ckpt, dict) and k in ckpt:
        v = ckpt[k]
        try:
            # Keep printable small values
            if isinstance(v, (int, float, str)):
                out[k] = v
            else:
                out[k] = {
                    "type": type(v).__name__,
                    "repr": str(v)[:300]
                }
        except Exception:
            out[k] = {"type": type(v).__name__}

# Try to extract epoch from common nested places
found = {}
if isinstance(ckpt, dict):
    if "epoch" in ckpt:
        found["epoch"] = ckpt["epoch"]
    # pytorch_lightning <=1.5 stores "callbacks" with ModelCheckpoint info
    cb = ckpt.get("callbacks")
    if isinstance(cb, dict):
        for name, val in cb.items():
            if "ModelCheckpoint" in name or "modelcheckpoint" in name.lower():
                # inspect for best_model_score or best_model_path
                if isinstance(val, dict):
                    if "best_model_score" in val:
                        found.setdefault("best_model_score", val["best_model_score"])
                    if "best_model_path" in val:
                        found.setdefault("best_model_path", val["best_model_path"])
    # sometimes "state_dict" has keys like "epoch" in nested trainer state
    trainer_state = ckpt.get("trainer") or ckpt.get("trainer_state")
    if isinstance(trainer_state, dict):
        if "global_step" in trainer_state:
            found.setdefault("global_step", trainer_state["global_step"]) 

# Fallback: global_step at top level
if isinstance(ckpt, dict) and "global_step" in ckpt:
    found.setdefault("global_step", ckpt["global_step"]) 

out["found"] = found
print(json.dumps(out, indent=2, default=str))
