import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_utils import build_dataset, PATCH_SIZE
from model import ECGAttentionClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pt")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

SEED = 0
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, split = build_dataset(seed=SEED)
    print(f"Dataset: {len(X)} beats | normal={int((y == 0).sum())} abnormal={int((y == 1).sum())}")

    X_trainval, y_trainval = X[split == "train"], y[split == "train"]
    X_test, y_test = X[split == "test"], y[split == "test"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
    )

    def make_loader(X_, y_, shuffle):
        ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, True)
    val_loader = make_loader(X_val, y_val, False)
    test_loader = make_loader(X_test, y_test, False)

    model = ECGAttentionClassifier(beat_len=X.shape[1], patch_size=PATCH_SIZE).to(device)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_loader.dataset)

        val_preds, val_targets = evaluate(model, val_loader, device)
        report = classification_report(val_targets, val_preds, output_dict=True, zero_division=0)
        val_f1 = report["macro avg"]["f1-score"]
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_acc={report['accuracy']:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state": model.state_dict(),
                "beat_len": int(X.shape[1]),
            }, MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_preds, test_targets = evaluate(model, test_loader, device)
    report = classification_report(test_targets, test_preds, target_names=["Normal", "Abnormal"], output_dict=True, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds).tolist()

    print("\nTest set report:")
    print(classification_report(test_targets, test_preds, target_names=["Normal", "Abnormal"], zero_division=0))
    print("Confusion matrix:", cm)

    with open(METRICS_PATH, "w") as f:
        json.dump({"test_report": report, "confusion_matrix": cm}, f, indent=2)
    print(f"\nSaved best model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(yb.numpy().tolist())
    return preds, targets


if __name__ == "__main__":
    main()
