import pandas as pd

def patch(path):
    df = pd.read_csv(path)
    if "_primary_label" not in df.columns:
        df["_primary_label"] = "unknown"

    df["_primary_label"] = df["behavior"].map({
        "ear": "cross_sucking",
        "tail": "cross_sucking",
        "teat": "cross_sucking",
        "other": "other",
    }).fillna("unknown")

    df.to_csv(path, index=False)

for p in ["data/manifests/train.csv", "data/manifests/val.csv"]:
    patch(p)

print("Patched _primary_label in train/val.")
