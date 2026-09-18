import json

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

cells = []

cells.append(md(
"""# EKG Kalp Atisi Siniflandirma - Attention Modeli (v2)

Bu surumde 5 gelistirme var:
1. **4 sinifli** AAMI siniflandirmasi (N/S/V/F), sadece Normal/Abnormal degil
2. **Hasta-bazli (patient-level) train/val/test bolme** - veri sizintisini onluyor
3. **Konvolusyon + Attention hibrit** model (once yerel desenler, sonra global attention)
4. **Cok katmanli** attention gorsellestirme (1. katman vs 2. katman)
5. **Head-bazli** attention gorsellestirme (her attention head'i ayri ayri)

**Ayarlar:** Sag panelden **Add Data** ile `mondejar/mitbih-database` veri
setini ekle, Settings > Accelerator > GPU sec (internete gerek yok). Sonra
Run All."""
))

cells.append(code(
"""import os, json, glob, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
"""
))

cells.append(md(
"""## 1. Veri: ham MIT-BIH kayitlarini yukle, atislara ayir

**Veri seti:** `mondejar/mitbih-database` (Kaggle) - internet gerektirmeyen,
her hasta/kayit icin ayri `<id>.csv` (ham sinyal) ve `<id>annotations.txt`
(R-peak konumlari + tip etiketleri) iceren bir MIT-BIH kopyasi. Bunu
notebooka **Add Data** ile eklemen gerekiyor (asagida kontrol ediyoruz).

Her atisi R-peak (kalp atisinin en belirgin sivri noktasi) etrafinda 256
ornekten olusan bir pencere olarak kesiyoruz (128 once + 128 sonra). Ayrica
her atisin **hangi kayittan (hastadan)** geldigini de saklıyoruz - bu bilgiyi
birazdan hasta-bazli train/test bolmede kullanacagiz."""
))

cells.append(code(
'''# Pace edilen (paced) kayitlar (102,104,107,217) haric tutuluyor: bunlarin
# "normal" atislari kalp pili kaynakli, diger atislarla karsilastirilabilir degil.
RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
    "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "219", "220", "221", "222", "223", "228", "230",
    "231", "232", "233", "234",
]

PRE, POST = 128, 128
BEAT_LEN = PRE + POST
PATCH_SIZE = 8  # 256 = 8 * 32 -> 32 token/atis

# AAMI 4 sinif: N=Normal, S=Supraventrikuler, V=Ventrikuler, F=Fuzyon
# Not: "Q" (bilinmeyen) sinifini disarida birakiyoruz - paced kayitlar zaten
# elenince bu veri setinde toplam 15 atisla temsil ediliyor, istatistiksel
# olarak ogrenilemeyecek kadar az. Bu yuzden 5 degil 4 sinifli calisiyoruz.
CLASS_NAMES = ["N", "S", "V", "F"]
SYMBOL_TO_CLASS = {}
for s in "NLRej":
    SYMBOL_TO_CLASS[s] = 0
for s in "AaJS":
    SYMBOL_TO_CLASS[s] = 1
for s in "VE":
    SYMBOL_TO_CLASS[s] = 2
for s in "F":
    SYMBOL_TO_CLASS[s] = 3

matches = glob.glob("/kaggle/input/**/100.csv", recursive=True)
if not matches:
    raise FileNotFoundError(
        "100.csv bulunamadi. Sag panelden 'Add Data' ile "
        "'MIT-BIH Database' (mondejar/mitbih-database) veri setini ekle."
    )
DATA_DIR = os.path.dirname(matches[0])
print("Veri klasoru bulundu:", DATA_DIR)

def load_record(rec_id):
    sig_df = pd.read_csv(os.path.join(DATA_DIR, f"{rec_id}.csv"))
    signal = sig_df.iloc[:, 1].to_numpy(dtype=np.float32)  # ilk sinyal kanali (genelde MLII)

    ann_df = pd.read_csv(
        os.path.join(DATA_DIR, f"{rec_id}annotations.txt"),
        sep=r"\\s+", skiprows=1, header=None, quoting=3, engine="python",
        names=["Time", "Sample", "Type", "Sub", "Chan", "Num", "Aux"],
        on_bad_lines="skip",
    )
    return signal, ann_df["Sample"].to_numpy(), ann_df["Type"].astype(str).to_numpy()

all_beats, all_labels, all_record_ids = [], [], []
for rec_id in RECORDS:
    signal, samples, symbols = load_record(rec_id)

    for sample_idx, symbol in zip(samples, symbols):
        if symbol not in SYMBOL_TO_CLASS:
            continue
        start, end = sample_idx - PRE, sample_idx + POST
        if start < 0 or end > len(signal):
            continue
        window = signal[start:end].astype(np.float32)
        std = window.std()
        window = (window - window.mean()) / std if std > 1e-6 else window - window.mean()

        all_beats.append(window)
        all_labels.append(SYMBOL_TO_CLASS[symbol])
        all_record_ids.append(rec_id)

    print(f"{rec_id}: toplam atis so far = {len(all_beats)}")

X = np.stack(all_beats).astype(np.float32)
y = np.array(all_labels, dtype=np.int64)
record_ids = np.array(all_record_ids)

print("\\nToplam atis:", len(X))
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(y == i).sum()}")
'''
))

cells.append(md(
"""## 2. Hasta-bazli (patient-level) train/val/test bolme

`GroupShuffleSplit` ile bolme yapiyoruz - "grup" olarak `record_ids` (kayit/hasta
numarasi) veriyoruz. Bu, ayni hastanin atislarinin hem train hem test'te
bulunmasini engelliyor. Ayrica egitim setinde Normal sinifinin asiri baskin
olmasini (class imbalance) sinirliyoruz."""
))

cells.append(code(
'''gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
trainval_idx, test_idx = next(gss1.split(X, y, groups=record_ids))

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)  # trainval'in %80'i train, %20'si val
train_idx_local, val_idx_local = next(gss2.split(X[trainval_idx], y[trainval_idx], groups=record_ids[trainval_idx]))
train_idx = trainval_idx[train_idx_local]
val_idx = trainval_idx[val_idx_local]

# Egitim setinde asiri baskin Normal sinifini 4:1 orani ile sinirla
rng = np.random.default_rng(SEED)
normal_idx = train_idx[y[train_idx] == 0]
other_idx = train_idx[y[train_idx] != 0]
max_normal = int(len(other_idx) * 4.0)
if len(normal_idx) > max_normal:
    normal_idx = rng.choice(normal_idx, size=max_normal, replace=False)
train_idx = np.sort(np.concatenate([normal_idx, other_idx]))

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train:", len(X_train), "hasta sayisi:", len(set(record_ids[train_idx])))
print("Val:  ", len(X_val), "hasta sayisi:", len(set(record_ids[val_idx])))
print("Test: ", len(X_test), "hasta sayisi:", len(set(record_ids[test_idx])))
assert set(record_ids[train_idx]) & set(record_ids[val_idx]) == set()
assert set(record_ids[train_idx]) & set(record_ids[test_idx]) == set()
print("Kontrol OK: hic hasta iki sette birden yok.")
'''
))

cells.append(md(
"""## 3. Model: Konvolusyon + Attention hibrit

Onceki surumde atisi direkt 8'erli parcalara (patch) bolup attention'a
veriyorduk. Bu sefer once kucuk bir **konvolusyon katmani** (`LocalConvStem`)
sinyal uzerinde kayarak yerel desenleri (kucuk dalgacik sekilleri) ogreniyor,
sonra bu islenmis sinyal parcalara bolunup **self-attention**'a giriyor. Boylece
model hem yerel (konvolusyon) hem uzun-menzilli (attention) iliskileri
yakalayabiliyor - literatürdeki CAT-Net gibi hibrit mimarilerin fikri budur."""
))

cells.append(code(
'''class LocalConvStem(nn.Module):
    """Patch'lere bolmeden once sinyalde yerel (kisa menzilli) desenleri ogrenir."""

    def __init__(self, hidden=16, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=kernel_size, padding=pad),
        )

    def forward(self, x):
        # x: (batch, 1, beat_len) -> (batch, 1, beat_len)
        return x + self.net(x)  # residual: yerel duzeltme ekliyor, ham sinyali kaybetmiyor


class AttentionBlock(nn.Module):
    """Self-attention + MLP blogu, attention agirliklarini da dondurur."""

    def __init__(self, d_model, n_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights  # (batch, n_heads, seq, seq)


class ECGAttentionClassifier(nn.Module):
    def __init__(self, beat_len=256, patch_size=8, d_model=64, n_heads=4, n_layers=2, n_classes=4, dropout=0.1):
        super().__init__()
        assert beat_len % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = beat_len // patch_size

        self.local_conv = LocalConvStem()
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([AttentionBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_attn=False):
        b = x.shape[0]
        x = self.local_conv(x.unsqueeze(1))          # (batch, 1, beat_len)
        tokens = self.patch_embed(x).transpose(1, 2)  # (batch, n_patches, d_model)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed

        attn_maps = []
        for block in self.blocks:
            tokens, attn_weights = block(tokens)
            attn_maps.append(attn_weights)  # her katman icin ayri sakliyoruz

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if return_attn:
            return logits, attn_maps
        return logits

    def cls_attention(self, attn_maps, layer_idx=-1, head_idx=None):
        """[CLS] token'in patch'lere attention agirligi.

        layer_idx: hangi katman (varsayilan: son katman)
        head_idx: None ise tum head'lerin ortalamasi, sayi verilirse o head
        """
        layer_attn = attn_maps[layer_idx]  # (batch, n_heads, seq, seq)
        if head_idx is None:
            cls_to_all = layer_attn[:, :, 0, :].mean(dim=1)
        else:
            cls_to_all = layer_attn[:, head_idx, 0, :]
        return cls_to_all[:, 1:]  # CLS->CLS kismini at, sadece patch'leri birak
'''
))

cells.append(md("## 4. Egitim"))

cells.append(code(
'''BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
N_CLASSES = len(CLASS_NAMES)

def make_loader(X_, y_, shuffle):
    ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val, False)
test_loader = make_loader(X_test, y_test, False)

model = ECGAttentionClassifier(beat_len=BEAT_LEN, patch_size=PATCH_SIZE, n_classes=N_CLASSES).to(device)
print(model)
print("Parametre sayisi:", sum(p.numel() for p in model.parameters()))

class_counts = np.bincount(y_train, minlength=N_CLASSES).clip(min=1)
class_weights = torch.tensor(len(y_train) / (N_CLASSES * class_counts), dtype=torch.float32).to(device)
print("Sinif agirliklari:", dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(2))))

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(yb.numpy().tolist())
    return preds, targets

best_val_f1 = -1.0
MODEL_PATH = "/kaggle/working/model.pt"

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

    val_preds, val_targets = evaluate(model, val_loader)
    report = classification_report(val_targets, val_preds, labels=list(range(N_CLASSES)), output_dict=True, zero_division=0)
    val_f1 = report["macro avg"]["f1-score"]
    print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_acc={report[\'accuracy\']:.4f} | val_macro_f1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({"model_state": model.state_dict(), "beat_len": BEAT_LEN, "patch_size": PATCH_SIZE, "n_classes": N_CLASSES}, MODEL_PATH)
'''
))

cells.append(md("## 5. Test sonuclari"))

cells.append(code(
'''checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])

test_preds, test_targets = evaluate(model, test_loader)
report = classification_report(test_targets, test_preds, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
cm = confusion_matrix(test_targets, test_preds, labels=list(range(N_CLASSES))).tolist()

print(classification_report(test_targets, test_preds, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES, zero_division=0))
print("Confusion matrix (satir=gercek, sutun=tahmin), siralama:", CLASS_NAMES)
print(np.array(cm))

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump({"test_report": report, "confusion_matrix": cm, "class_names": CLASS_NAMES}, f, indent=2)
'''
))

cells.append(md(
"""## 6. Cok katmanli + Head-bazli attention gorsellestirme

Modelimizde **2 attention katmani** ve her katmanda **4 head (kafa)** var.
Simdiye kadar bunlarin hepsinin ortalamasini aliyorduk. Bu sefer:
- **Katmanlar arasi karsilastirma:** 1. katman ile 2. katman farkli yerlere mi bakiyor?
- **Head'ler arasi karsilastirma:** Ayni katmandaki 4 head birbirinden farkli
  ozelliklere mi uzmanlasmis?

Bu, transformer modellerinin "kara kutu olmadigini", farkli katman ve
head'lerin farkli seyler ogrenebildigini gormemizi sagliyor."""
))

cells.append(code(
'''x_tensor = torch.from_numpy(X_test).to(device)
model.eval()
with torch.no_grad():
    logits, attn_maps = model(x_tensor, return_attn=True)
preds = logits.argmax(dim=1).cpu().numpy()
n_layers = len(attn_maps)
n_heads = attn_maps[0].shape[1]
print(f"Katman sayisi: {n_layers}, head sayisi: {n_heads}")

def per_sample_curve(cls_attn_row, patch_size, beat_len):
    curve = np.repeat(cls_attn_row, patch_size)[:beat_len]
    return (curve - curve.min()) / (np.ptp(curve) + 1e-8)

def plot_heat(ax, beat, attn_row, patch_size, title):
    beat_len = len(beat)
    curve = per_sample_curve(attn_row, patch_size, beat_len)
    x = np.arange(beat_len)
    ax.pcolormesh(x, [0, 1], np.tile(curve, (2, 1)), cmap="Reds", alpha=0.6,
                  shading="auto", transform=ax.get_xaxis_transform(), vmin=0, vmax=1)
    ax.plot(x, beat, color="black", linewidth=1.0)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, beat_len - 1)
    ax.set_yticks([]); ax.set_xticks([])

# Her siniftan dogru tahmin edilen birer ornek sec (varsa)
example_idx = {}
for c in range(N_CLASSES):
    matches = np.where((preds == y_test) & (y_test == c))[0]
    if len(matches) > 0:
        example_idx[c] = matches[0]

print("Ornek bulunan siniflar:", [CLASS_NAMES[c] for c in example_idx])

for c, idx in example_idx.items():
    beat = X_test[idx]
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 2.2 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]
    for layer in range(n_layers):
        for head in range(n_heads):
            attn_row = model.cls_attention(attn_maps, layer_idx=layer, head_idx=head)[idx].cpu().numpy()
            plot_heat(axes[layer, head], beat, attn_row, PATCH_SIZE,
                      f"Katman {layer+1}, Head {head+1}")
    fig.suptitle(f"Sinif: {CLASS_NAMES[c]} (true=pred={CLASS_NAMES[c]}) - katman x head attention", y=1.02)
    fig.tight_layout()
    fig.savefig(f"/kaggle/working/attention_layers_heads_{CLASS_NAMES[c]}.png", dpi=130, bbox_inches="tight")
    plt.show()
'''
))

cells.append(md(
"""## 7. Katman-ortalamali karsilastirma (ozet gorsel)

Her katmanin (head'ler ortalanmis) tum ornekler icin genel attention
davranisini tek bir grafikte karsilastiralim."""
))

cells.append(code(
'''fig, axes = plt.subplots(len(example_idx), 1, figsize=(9, 2.2 * len(example_idx)))
if len(example_idx) == 1:
    axes = [axes]

for ax, (c, idx) in zip(axes, example_idx.items()):
    beat = X_test[idx]
    beat_len = len(beat)
    x = np.arange(beat_len)
    ax.plot(x, beat, color="black", linewidth=1.0, label="EKG")
    for layer in range(n_layers):
        attn_row = model.cls_attention(attn_maps, layer_idx=layer, head_idx=None)[idx].cpu().numpy()
        curve = per_sample_curve(attn_row, PATCH_SIZE, beat_len)
        ax.plot(x, curve, linewidth=1.5, label=f"Katman {layer+1} attention (0-1 olcekli)")
    ax.set_title(f"Sinif: {CLASS_NAMES[c]}", fontsize=10)
    ax.set_xlim(0, beat_len - 1)
    ax.legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("Atis icindeki ornek indeksi (R-peak merkezli)")
fig.suptitle("Katmanlar arasi attention karsilastirmasi", y=1.0)
fig.tight_layout()
fig.savefig("/kaggle/working/attention_layer_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
'''
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("notebook.ipynb yazildi")
