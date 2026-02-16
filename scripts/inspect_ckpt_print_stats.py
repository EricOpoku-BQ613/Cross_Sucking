"""
Print `train_stats` and `val_stats` from a PyTorch checkpoint.
Usage: python scripts/inspect_ckpt_print_stats.py <ckpt_path>
"""
import sys, json
path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    print("Usage: script <ckpt_path>")
    sys.exit(2)
import torch
ckpt = torch.load(path, map_location='cpu')
out = {}
for k in ('train_stats','val_stats','best_metric','epoch'):
    if k in ckpt:
        out[k] = ckpt[k]
print(json.dumps(out, indent=2, default=str))
