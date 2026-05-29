import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def _build_efficientnet_features():
    try:
        net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        net = models.efficientnet_b0(pretrained=True)
    return net.features


class SliceTransformer(nn.Module):
    """Transformer encoder mô hình hóa quan hệ giữa các slice MRI theo chiều sâu.

    Thay vì max-pooling theo slice như MRNet gốc, module này dùng
    self-attention để học tầm quan trọng của từng slice.
    """

    def __init__(self, d_model: int = 1280, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1, max_slices: int = 64):
        super().__init__()
        # CLS token đại diện cho toàn bộ chuỗi slice
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embedding học được (không cần sinusoidal vì thứ tự slice quan trọng)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm: ổn định hơn khi fine-tune
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] — batch, số slice, feature dim
        B, S, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                  # [B, S+1, D]
        x = x + self.pos_embed[:, : S + 1, :]
        x = self.encoder(x)
        return self.norm(x[:, 0])                        # CLS token: [B, D]


class EfficientNetViT(nn.Module):
    """EfficientNetB0 + ViT Transformer cho phân loại MRI 3 mặt phẳng.

    Kiến trúc:
      - EfficientNetB0 (pretrained): trích xuất đặc trưng không gian từng slice
      - SliceTransformer (ViT encoder): mô hình hóa quan hệ giữa các slice
      - Ghép 3 mặt phẳng → MLP classifier → 1 logit (binary)

    Luồng xử lý mỗi mặt phẳng:
      [B, S, 3, H, W] → EfficientNetB0 → AdaptivePool → [B, S, 1280]
                      → SliceTransformer → [B, 1280]  (CLS token)
    """

    def __init__(
        self,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        max_slices: int = 64,
    ):
        super().__init__()
        feat_dim = 1280

        self.axial_backbone = _build_efficientnet_features()
        self.coronal_backbone = _build_efficientnet_features()
        self.sagittal_backbone = _build_efficientnet_features()

        if freeze_backbone:
            self.freeze_backbones()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.axial_vit = SliceTransformer(feat_dim, nhead, num_layers, dropout, max_slices)
        self.coronal_vit = SliceTransformer(feat_dim, nhead, num_layers, dropout, max_slices)
        self.sagittal_vit = SliceTransformer(feat_dim, nhead, num_layers, dropout, max_slices)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1),
        )

    def freeze_backbones(self):
        """Đóng băng toàn bộ EfficientNetB0 backbone (chỉ train transformer)."""
        for net in [self.axial_backbone, self.coronal_backbone, self.sagittal_backbone]:
            for p in net.parameters():
                p.requires_grad = False

    def unfreeze_backbones(self):
        """Mở đóng băng để fine-tune toàn bộ model."""
        for net in [self.axial_backbone, self.coronal_backbone, self.sagittal_backbone]:
            for p in net.parameters():
                p.requires_grad = True

    def _encode_plane(self, backbone, vit, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [S, 3, H, W] — khi batch_size=1 và đã squeeze
            S = x.shape[0]
            feats = backbone(x)                          # [S, 1280, h, w]
            feats = self.pool(feats).view(1, S, -1)      # [1, S, 1280]
        elif x.dim() == 5:
            # [B, S, 3, H, W]
            B, S, C, H, W = x.shape
            feats = backbone(x.view(B * S, C, H, W))    # [B*S, 1280, h, w]
            feats = self.pool(feats).view(B, S, -1)      # [B, S, 1280]
        else:
            raise ValueError(f"Shape không hợp lệ: {x.shape}")
        return vit(feats)                                # [B, 1280]

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("Input phải là list/tuple gồm 3 tensor [axial, coronal, sagittal].")

        axial, coronal, sagittal = x
        f_a = self._encode_plane(self.axial_backbone, self.axial_vit, axial)
        f_c = self._encode_plane(self.coronal_backbone, self.coronal_vit, coronal)
        f_s = self._encode_plane(self.sagittal_backbone, self.sagittal_vit, sagittal)

        combined = torch.cat([f_a, f_c, f_s], dim=1)   # [B, 3840]
        return self.classifier(combined)                 # [B, 1]
