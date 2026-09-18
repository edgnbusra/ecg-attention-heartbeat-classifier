# ECG Attention Heartbeat Classifier

A small Transformer-style attention model that classifies individual heartbeats from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) into AAMI classes
(N / S / V / F), with attention-weight visualization to inspect which part of the waveform
the model relies on.

**Full write-up (Kaggle notebook, results, patient-level split, figures):**
https://www.kaggle.com/code/hffjbrbbgmailcom/ecg-attention-heartbeat-classifier

## Idea

Each heartbeat (a 256-sample window centered on its R-peak) is split into 32 patches of 8
samples each — analogous to tokens in NLP. A small local convolution stem extracts local
waveform shapes, patches are embedded into 64-dim vectors, and a learnable `[CLS]` token
attends over the patches through a 2-layer, 4-head self-attention encoder. Classification is
read off the `[CLS]` token; its attention weights over patches double as an interpretability
signal, visualized as a heatmap over the ECG waveform.

## Key design choices

- **Patient-level train/val/test split** (`GroupShuffleSplit`, grouped by MIT-BIH record id)
  to avoid data leakage — beats from the same patient never appear in both train and test.
- **4-class AAMI labeling** (N, S, V, F); the Q class was dropped (only 15 beats across the
  44 non-paced records used — too few to learn).
- **Conv + attention hybrid**: a residual local-conv block before patch embedding, combining
  local pattern extraction with global self-attention (inspired by CAT-Net-style hybrids).
- **Per-layer / per-head attention visualization**, not just an averaged heatmap.

## Results (patient-level test set, 14 held-out patients)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| N (Normal) | 0.92 | 0.68 | 0.78 | 27,204 |
| S (Supraventricular) | 0.12 | 0.38 | 0.18 | 2,042 |
| V (Ventricular) | 0.44 | 0.80 | 0.56 | 3,106 |
| F (Fusion) | 0.005 | 0.003 | 0.003 | 393 |
| **Accuracy** | | | **0.67** | 32,745 |
| **Macro F1** | | | **0.38** | |

Note: an earlier, beat-level-random-split version of this project reached 97% accuracy on a
binary Normal/Abnormal task. That number was inflated by patient-level data leakage (the same
patient's beats appearing in both train and test); the numbers above, from a proper
patient-level split and a harder 4-class task, are the honest estimate. See the Kaggle
notebook for the full discussion.

### Attention visualization

The `[CLS]` token consistently attends to the QRS complex / R-peak region across all four
classes, with the first attention layer spreading more broadly (notably for the V class,
matching the wider, distorted QRS morphology of ventricular beats) and the second layer
sharpening onto the R-peak itself. See `outputs/kaggle_run/attention_layer_comparison.png`
and `outputs/kaggle_run/attention_layers_heads_*.png`.

## Repo structure

```
src/                  Local reference implementation (data loading, model, train, visualize)
kaggle_notebook/       Source of truth: the notebook actually trained on Kaggle (GPU)
  build_notebook.py    Generates notebook.ipynb programmatically
  notebook.ipynb        Generated notebook
  kernel-metadata.json  Kaggle kernel config (dataset, GPU, etc.)
outputs/kaggle_run/    Downloaded results: metrics.json, trained model, attention figures
```

Raw data (MIT-BIH CSVs) is not included in this repo — see the notebook's data-loading cell
for the Kaggle dataset used (`mondejar/mitbih-database`).

## Limitations & next steps

- Only 44 patients total; after patient-level splitting, rare classes (S, F) have very little
  training diversity — F is essentially unlearned (recall ≈ 0).
- Single train/val/test split; results should be confirmed with patient-level k-fold
  cross-validation.
- Not clinically validated — this is a learning project, not a diagnostic tool.

---

## Türkçe Özet

MIT-BIH Arrhythmia Database üzerinden, her kalp atışını küçük zaman dilimlerine (token) bölüp
self-attention ile Normal/Supraventriküler/Ventriküler/Füzyon (AAMI N/S/V/F) sınıflandırması
yapan küçük bir Transformer modeli. Hasta-bazlı (patient-level) train/test bölmesiyle veri
sızıntısı engellenmiş, sonuçlar bu nedenle daha düşük ama daha gerçekçidir (%67 doğruluk,
makro-F1 0.38). Attention ağırlıkları, modelin QRS/R-peak bölgesine tutarlı şekilde
odaklandığını gösteriyor. Detaylı analiz ve tartışma için yukarıdaki Kaggle notebook linkine
bakabilirsiniz.
