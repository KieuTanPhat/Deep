import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class _SliceAttentionPool(nn.Module):
    """Học trọng số tầm quan trọng của từng slice thay vì max-pool cứng."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        weights = torch.softmax(self.gate(x), dim=1)  # [B, S, 1]
        return (x * weights).sum(dim=1)               # [B, D]


class _CrossPlaneTransformer(nn.Module):
    """Một Transformer layer để axial/coronal/sagittal tương tác với nhau."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, D] — Pre-LN self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class EfficientNetB0_ViT(nn.Module):
    """
    EfficientNetB0 + Vision Transformer nhẹ cho phân loại MRI 3 mặt phẳng.

    Cải tiến so với EfficientNetB0 thuần:
      1. Intra-slice ViT  — self-attention không gian trong từng slice,
                            học global context mà CNN không nắm được.
      2. Attention slice pooling — học slice nào mang thông tin chẩn đoán,
                                   thay vì max-pool cứng nhắc.
      3. Cross-plane Transformer — mô hình hoá mối quan hệ axial/coronal/sagittal.

    Tương thích warm-start: dùng cùng tên key (self.axial/coronal/sagittal)
    với EfficientNetB0 gốc nên checkpoint abnormal load được backbone CNN.
    """

    D_MODEL: int = 256   # chiều projection — nhẹ nhưng đủ biểu diễn
    N_HEADS: int = 4     # D_MODEL / N_HEADS = 64 (head dim tiêu chuẩn)
    N_LAYERS: int = 2    # độ sâu Transformer intra-slice

    def __init__(self):
        super().__init__()

        # Backbone CNN riêng cho từng mặt phẳng — giữ tên key giống
        # EfficientNetB0 gốc để warm-start từ checkpoint abnormal.
        self.axial    = _build_efficientnet_b0().features
        self.coronal  = _build_efficientnet_b0().features
        self.sagittal = _build_efficientnet_b0().features

        # Projection 1280 → D_MODEL (dùng chung cho cả 3 mặt phẳng)
        self.input_proj = nn.Sequential(
            nn.Linear(1280, self.D_MODEL),
            nn.LayerNorm(self.D_MODEL),
        )

        # Positional embedding không gian học được (EfficientNetB0-224 → 7×7 = 49 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, self.D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Intra-slice Transformer (dùng chung cho cả 3 mặt phẳng)
        vit_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=self.N_HEADS,
            dim_feedforward=self.D_MODEL * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN ổn định hơn khi train
        )
        self.vit      = nn.TransformerEncoder(vit_layer, num_layers=self.N_LAYERS, enable_nested_tensor=False)
        self.vit_norm = nn.LayerNorm(self.D_MODEL)

        # Attention pooling qua các slice
        self.slice_pool = _SliceAttentionPool(self.D_MODEL)

        # Cross-plane Transformer
        self.cross_plane = _CrossPlaneTransformer(self.D_MODEL, n_heads=4)

        # Classifier
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(3 * self.D_MODEL, 1)

    def _encode_plane(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Mã hoá một mặt phẳng MRI qua CNN → ViT → attention pooling slice.

        Args:
            net: EfficientNetB0 feature extractor
            x:   [S, C, H, W]  hoặc  [B, S, C, H, W]
        Returns:
            plane feature: [B, D_MODEL]  (B=1 khi input 4-D)
        """
        if x.dim() == 4:       # mẫu đơn — thêm batch dim
            x = x.unsqueeze(0)

        if x.dim() != 5:
            raise ValueError(f"Unexpected plane shape: {x.shape}")

        b, s, c, h, w = x.shape
        x_flat = x.view(b * s, c, h, w)

        # CNN: [B*S, 1280, 7, 7]
        feat_map = net(x_flat)

        # Flatten không gian → patch tokens: [B*S, n_patches, 1280]
        tokens = feat_map.flatten(2).transpose(1, 2)
        n_patches = tokens.size(1)

        # Projection + positional embedding: [B*S, n_patches, D_MODEL]
        tokens = self.input_proj(tokens)
        tokens = tokens + self.pos_embed[:, :n_patches, :]

        # Intra-slice Transformer: [B*S, n_patches, D_MODEL]
        tokens = self.vit(tokens)
        tokens = self.vit_norm(tokens)

        # Mean-pool patches → vector mỗi slice: [B*S, D_MODEL]
        slice_feat = tokens.mean(dim=1)
        slice_feat = slice_feat.view(b, s, self.D_MODEL)   # [B, S, D_MODEL]

        # Attention pooling qua các slice → vector mặt phẳng: [B, D_MODEL]
        return self.slice_pool(slice_feat)

    def forward(self, x):
        """
        Args:
            x: list 3 tensor [axial, coronal, sagittal]
               mỗi cái: [B, S, C, H, W]  hoặc  [S, C, H, W]
        Returns:
            logits: [B, 1]
        """
        axial    = self._encode_plane(self.axial,    x[0])   # [B, D_MODEL]
        coronal  = self._encode_plane(self.coronal,  x[1])
        sagittal = self._encode_plane(self.sagittal, x[2])

        # Cross-plane Transformer: [B, 3, D_MODEL]
        planes = torch.stack([axial, coronal, sagittal], dim=1)
        planes = self.cross_plane(planes)

        # Phân loại: [B, 3*D_MODEL] → [B, 1]
        feats = self.dropout(planes.flatten(1))
        return self.fc(feats)
