"""Visualize the model's [CLS]-to-patch attention weights over ECG beat waveforms."""
import os

import numpy as np
import torch

from data_utils import build_dataset, PATCH_SIZE
from model import ECGAttentionClassifier
from train import MODEL_PATH, SEED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "attention_plots")
LABELS = {0: "Normal", 1: "Abnormal"}


def load_test_split(X, y, split):
    return X[split == "test"], y[split == "test"]


def plot_beat_attention(ax, beat, patch_attn, patch_size, title):
    beat_len = len(beat)
    per_sample_attn = np.repeat(patch_attn, patch_size)[:beat_len]
    per_sample_attn = (per_sample_attn - per_sample_attn.min()) / (np.ptp(per_sample_attn) + 1e-8)

    x = np.arange(beat_len)
    ax.pcolormesh(
        x, [0, 1], np.tile(per_sample_attn, (2, 1)),
        cmap="Reds", alpha=0.55, shading="auto",
        transform=ax.get_xaxis_transform(), vmin=0, vmax=1,
    )
    ax.plot(x, beat, color="black", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, beat_len - 1)
    ax.set_yticks([])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(PLOT_DIR, exist_ok=True)

    X, y, split = build_dataset(seed=SEED)
    X_test, y_test = load_test_split(X, y, split)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = ECGAttentionClassifier(beat_len=checkpoint["beat_len"], patch_size=PATCH_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    x_tensor = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        logits, attn_maps = model(x_tensor, return_attn=True)
        patch_attn = model.cls_attention_over_patches(attn_maps).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()

    correct_normal = np.where((preds == y_test) & (y_test == 0))[0]
    correct_abnormal = np.where((preds == y_test) & (y_test == 1))[0]
    misclassified = np.where(preds != y_test)[0]

    rng = np.random.default_rng(SEED)
    picks = []
    picks += [(i, "Correctly classified: Normal") for i in rng.choice(correct_normal, size=min(3, len(correct_normal)), replace=False)]
    picks += [(i, "Correctly classified: Abnormal") for i in rng.choice(correct_abnormal, size=min(3, len(correct_abnormal)), replace=False)]
    if len(misclassified) > 0:
        picks += [(i, "Misclassified") for i in rng.choice(misclassified, size=min(2, len(misclassified)), replace=False)]

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.0 * len(picks)))
    if len(picks) == 1:
        axes = [axes]

    for ax, (idx, tag) in zip(axes, picks):
        true_label = LABELS[int(y_test[idx])]
        pred_label = LABELS[int(preds[idx])]
        title = f"{tag} | true={true_label}, pred={pred_label}"
        plot_beat_attention(ax, X_test[idx], patch_attn[idx], model.patch_size, title)

    axes[-1].set_xlabel("Sample index within beat (R-peak centered)")
    fig.suptitle("[CLS] attention weights over ECG beat (red = higher attention)", y=1.0)
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "attention_examples.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention visualization to {out_path}")


if __name__ == "__main__":
    main()
