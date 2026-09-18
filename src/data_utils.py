"""Load pre-extracted MIT-BIH heartbeats and prepare them for classification.

Uses the "ECG Heartbeat Categorization Dataset" (Kaggle: shayanfazeli/heartbeat),
which already segments each beat from the raw MIT-BIH signal into a fixed-length,
zero-padded window (187 samples) with an AAMI class label. This is much faster
and simpler than re-extracting beats from the raw WFDB records ourselves.

Expected files (any one of these locations is checked, in order):
  - /kaggle/input/heartbeat/mitbih_train.csv, mitbih_test.csv   (Kaggle notebook)
  - ./data/mitbih_train.csv, ./data/mitbih_test.csv             (local download)
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_PATH = os.path.join(DATA_DIR, "mitbih_beats.npz")

CANDIDATE_DIRS = [
    "/kaggle/input/heartbeat",
    DATA_DIR,
]

BEAT_LEN = 187
PATCH_SIZE = 11  # 187 = 11 * 17, divides evenly into 17 patch tokens

# Original dataset labels (AAMI classes): 0=N(normal), 1=S, 2=V, 3=F, 4=Q
# We collapse to binary: 0=Normal, 1=Abnormal
NORMAL_CLASS = 0.0


def _find_csv_dir() -> str:
    for d in CANDIDATE_DIRS:
        if os.path.exists(os.path.join(d, "mitbih_train.csv")) and os.path.exists(os.path.join(d, "mitbih_test.csv")):
            return d
    raise FileNotFoundError(
        "Could not find mitbih_train.csv / mitbih_test.csv.\n"
        "Download the 'ECG Heartbeat Categorization Dataset' from Kaggle "
        "(shayanfazeli/heartbeat) and place mitbih_train.csv and mitbih_test.csv "
        f"into: {DATA_DIR}"
    )


def build_dataset(max_normal_ratio: float = 4.0, seed: int = 0):
    """Load beats, cache to disk as a combined train+test set, return X, y, split."""
    if os.path.exists(PROCESSED_PATH):
        data = np.load(PROCESSED_PATH)
        return data["X"], data["y"], data["split"]

    csv_dir = _find_csv_dir()
    train_df = pd.read_csv(os.path.join(csv_dir, "mitbih_train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(csv_dir, "mitbih_test.csv"), header=None)

    def to_xy(df, split_name):
        signals = df.iloc[:, :BEAT_LEN].to_numpy(dtype=np.float32)
        classes = df.iloc[:, BEAT_LEN].to_numpy(dtype=np.float32)
        labels = (classes != NORMAL_CLASS).astype(np.int64)  # 0=Normal, 1=Abnormal
        splits = np.full(len(df), split_name)
        return signals, labels, splits

    X_train, y_train, split_train = to_xy(train_df, "train")
    X_test, y_test, split_test = to_xy(test_df, "test")

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    split = np.concatenate([split_train, split_test])

    # Cap class imbalance within the train split only (test split stays untouched
    # so evaluation reflects the real-world class distribution).
    rng = np.random.default_rng(seed)
    train_mask = split == "train"
    normal_idx = np.where(train_mask & (y == 0))[0]
    abnormal_idx = np.where(train_mask & (y == 1))[0]
    max_normal = int(len(abnormal_idx) * max_normal_ratio)
    if len(normal_idx) > max_normal:
        drop = rng.choice(normal_idx, size=len(normal_idx) - max_normal, replace=False)
        keep_mask = np.ones(len(X), dtype=bool)
        keep_mask[drop] = False
        X, y, split = X[keep_mask], y[keep_mask], split[keep_mask]

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(PROCESSED_PATH, X=X, y=y, split=split)
    print(f"Saved {len(X)} beats ({(y == 0).sum()} normal, {(y == 1).sum()} abnormal) to {PROCESSED_PATH}")
    return X, y, split


if __name__ == "__main__":
    X, y, split = build_dataset()
    print(X.shape, y.shape)
    print("Normal:", (y == 0).sum(), "Abnormal:", (y == 1).sum())
    print("Train:", (split == "train").sum(), "Test:", (split == "test").sum())
