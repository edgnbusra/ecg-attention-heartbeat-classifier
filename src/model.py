"""Small Transformer-style attention classifier for single-beat ECG windows.

Each beat (256 raw samples) is chopped into fixed-length patches that act as
"tokens". A learnable [CLS] token attends over the patch tokens; its final
attention weights show which parts of the beat the model relied on.
"""
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Self-attention + MLP block that also returns its attention weights."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights  # attn_weights: (batch, n_heads, seq, seq)


class ECGAttentionClassifier(nn.Module):
    def __init__(self, beat_len: int = 256, patch_size: int = 8, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        assert beat_len % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = beat_len // patch_size

        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # x: (batch, beat_len)
        b = x.shape[0]
        tokens = self.patch_embed(x.unsqueeze(1)).transpose(1, 2)  # (batch, n_patches, d_model)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed

        attn_maps = []
        for block in self.blocks:
            tokens, attn_weights = block(tokens)
            attn_maps.append(attn_weights)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])  # classify from CLS token

        if return_attn:
            return logits, attn_maps
        return logits

    def cls_attention_over_patches(self, attn_maps):
        """Average the last block's CLS->patch attention over heads.

        Returns: (batch, n_patches) attention weight per patch.
        """
        last = attn_maps[-1]  # (batch, n_heads, seq, seq)
        cls_to_all = last[:, :, 0, :].mean(dim=1)  # (batch, seq) averaged over heads
        return cls_to_all[:, 1:]  # drop attention-to-self(CLS), keep patches
