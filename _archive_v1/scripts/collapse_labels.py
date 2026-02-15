from pathlib import Path
import pandas as pd

# collapse rule
KEEP = {"ear", "tail"}
OTHER_NAME = "other"  # merged bucket

def collapse_behavior(b: str) -> str:
    b = str(b).strip().lower()
    if b in KEEP:
        return b
    return OTHER_NAME

def process_one(csv_in: Path, csv_out: Path):
    df = pd.read_csv(csv_in)
    if "behavior" not in df.columns:
        raise ValueError(f"{csv_in} missing 'behavior'")
    df["behavior"] = df["behavior"].map(collapse_behavior)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"[OK] wrote {csv_out} | counts:\n{df['behavior'].value_counts(dropna=False)}\n")

def main():
    base = Path("data/manifests")
    for name in ["train", "val", "test"]:
        process_one(base / f"{name}.csv", base / f"{name}_collapsed.csv")

if __name__ == "__main__":
    main()
