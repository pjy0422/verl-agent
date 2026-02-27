#!/usr/bin/env python3
"""
Prepare jailbreak behavior dataset for multiturn_conv environment.

Train data: walledai/AdvBench (prompt field) from HuggingFace
Val data: harmbench_behaviors_text_all.csv (Behavior field) local CSV
"""
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
HARMBENCH_CSV = SCRIPT_DIR / "harmbench_behaviors_text_all.csv"


def create_entries(behaviors, split):
    """Convert a list of behavior strings into dataset entries."""
    data = []
    for idx, behavior in enumerate(behaviors):
        entry = {
            "data_source": "jailbreak",
            "prompt": [{
                "role": "user",
                "content": "",  # Initial prompt is empty for multiturn
            }],
            "ability": "agent",
            "behavior": behavior,
            "extra_info": {
                "split": split,
                "index": idx,
                "behavior_category": "harmful"
            }
        }
        data.append(entry)
    return pd.DataFrame(data)


def create_train_dataset(train_size: int = 520):
    """Load train behaviors from walledai/AdvBench (prompt field)."""
    ds = load_dataset("walledai/AdvBench", split="train")
    behaviors = [row["prompt"] for row in ds]
    behaviors = behaviors[:None if train_size == -1 else train_size]
    return create_entries(behaviors, "train")


def create_val_dataset(val_size: int = 400, csv_path: Path = HARMBENCH_CSV):
    """Load val behaviors from harmbench_behaviors_text_all.csv (Behavior field)."""
    df = pd.read_csv(csv_path)
    behaviors = df["Behavior"].tolist()
    behaviors = behaviors[:val_size]
    return create_entries(behaviors, "test")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare jailbreak dataset for multiturn_conv environment"
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/verl-agent/jailbreak",
        help="Local directory to save the dataset"
    )
    parser.add_argument(
        "--train_data_size",
        default=520,
        type=int,
        help="Number of training examples"
    )
    parser.add_argument(
        "--val_data_size",
        default=400,
        type=int,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--harmbench_csv",
        default=str(HARMBENCH_CSV),
        help="Path to harmbench_behaviors_text_all.csv"
    )

    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating jailbreak dataset...")
    print(f"  Train size: {args.train_data_size}")
    print(f"  Val size: {args.val_data_size}")
    print(f"  Output dir: {local_dir}")

    # Train: from walledai/AdvBench
    print("\nLoading train data from walledai/AdvBench ...")
    train_df = create_train_dataset(args.train_data_size)

    # Val: from harmbench CSV
    print(f"Loading val data from {args.harmbench_csv} ...")
    val_df = create_val_dataset(args.val_data_size, Path(args.harmbench_csv))

    # Save to parquet
    train_path = local_dir / "train.parquet"
    val_path = local_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved training data ({len(train_df)} rows) to: {train_path}")
    print(f"Saved validation data ({len(val_df)} rows) to: {val_path}")
    print(f"\nTrain sample:")
    print(train_df.iloc[0].to_dict())
    print(f"\nVal sample:")
    print(val_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
