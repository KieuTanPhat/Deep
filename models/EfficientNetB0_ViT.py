import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


def _trunc_normal_(tensor, std=0.02):
    init_fn = getattr(nn.init, "trunc_normal_", nn.init.normal_)
    init_fn(tensor, std=std)


class _SliceAttentionPool(nn.Module):
    """Learn a soft importance weight for each MRI slice."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        return torch.sum(x * weights, dim=1)


class EfficientNetB0_ViT(nn.Module):
    """EfficientNet-B0 + ViT-style MRI classifier.

    This version keeps the useful pieces from the existing target repo model
    and adds ordered slice tokens:

    - EfficientNet-B0 extracts per-slice feature maps.
    - A light patch transformer models within-slice spatial context.
    - A slice transformer with learned positional embeddings models MRI depth.
    - A cross-plane transformer models axial/coronal/sagittal interactions.

    The backbones are named axial/coronal/sagittal so EfficientNetB0 checkpoints
    can still warm-start the CNN feature extractors.
    """

    def __init__(
        self,
        feature_dim=1280,
        vit_dim=256,
        vit_depth=2,
        vit_heads=4,
        vit_mlp_ratio=2.0,
        vit_dropout=0.15,
        classifier_dropout=0.35,
        max_slices=64,
        pooling="cls_attention",
        patch_depth=1,
        patch_tokens=49,
    ):
        super().__init__()
        vit_dim = int(vit_dim)
        vit_heads = int(vit_heads)
        if vit_dim % vit_heads != 0:
            raise ValueError("vit_dim must be divisible by vit_heads.")

        self.feature_dim = int(feature_dim)
        self.vit_dim = vit_dim
        self.max_slices = max(1, int(max_slices))
        self.pooling = pooling
        self.patch_tokens = int(patch_tokens)

        valid_pooling = {"cls", "mean", "max", "attention", "cls_attention"}
        if self.pooling not in valid_pooling:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        self.axial = _build_efficientnet_b0().features
        self.coronal = _build_efficientnet_b0().features
        self.sagittal = _build_efficientnet_b0().features

        self.patch_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.vit_dim),
            nn.LayerNorm(self.vit_dim),
        )
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, self.patch_tokens, self.vit_dim))

        patch_layers = []
        for _ in range(max(0, int(patch_depth))):
            patch_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.vit_dim,
                    nhead=vit_heads,
                    dim_feedforward=int(self.vit_dim * float(vit_mlp_ratio)),
                    dropout=float(vit_dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
        self.patch_encoder = nn.TransformerEncoder(
            patch_layers[0],
            num_layers=len(patch_layers),
            enable_nested_tensor=False,
        ) if patch_layers else nn.Identity()
        self.patch_norm = nn.LayerNorm(self.vit_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_dim))
        self.slice_pos_embed = nn.Parameter(torch.zeros(1, self.max_slices, self.vit_dim))

        slice_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim,
            nhead=vit_heads,
            dim_feedforward=int(self.vit_dim * float(vit_mlp_ratio)),
            dropout=float(vit_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slice_encoder = nn.TransformerEncoder(
            slice_layer,
            num_layers=int(vit_depth),
            enable_nested_tensor=False,
        )
        self.slice_norm = nn.LayerNorm(self.vit_dim)
        self.slice_attn_pool = _SliceAttentionPool(self.vit_dim)
        self.cls_attn_fuse = nn.Linear(self.vit_dim * 2, self.vit_dim)

        self.plane_embed = nn.Parameter(torch.zeros(1, 3, self.vit_dim))
        plane_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim,
            nhead=vit_heads,
            dim_feedforward=int(self.vit_dim * float(vit_mlp_ratio)),
            dropout=float(vit_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.plane_encoder = nn.TransformerEncoder(
            plane_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(3 * self.vit_dim),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(3 * self.vit_dim, self.vit_dim),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(self.vit_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        _trunc_normal_(self.patch_pos_embed)
        _trunc_normal_(self.cls_token)
        _trunc_normal_(self.slice_pos_embed)
        _trunc_normal_(self.plane_embed)

    def freeze_feature_extractors(self, freeze=True):
        for parameter in self.backbone_parameters():
            parameter.requires_grad = not freeze

    def backbone_parameters(self):
        for module in (self.axial, self.coronal, self.sagittal):
            for parameter in module.parameters():
                yield parameter

    def head_parameters(self):
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in backbone_ids:
                yield parameter

    def _interpolate_pos(self, pos_embed, length):
        if length <= pos_embed.size(1):
            return pos_embed[:, :length, :]
        pos = pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=length, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def _pool_slice_tokens(self, tokens):
        cls_feat = tokens[:, 0, :]
        slice_tokens = tokens[:, 1:, :]

        if self.pooling == "cls":
            return cls_feat
        if self.pooling == "mean":
            return slice_tokens.mean(dim=1)
        if self.pooling == "max":
            return slice_tokens.max(dim=1)[0]

        attn_feat = self.slice_attn_pool(slice_tokens)
        if self.pooling == "attention":
            return attn_feat
        return self.cls_attn_fuse(torch.cat([cls_feat, attn_feat], dim=1))

    def _encode_plane(self, net, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.dim() != 5:
            raise ValueError(f"Unexpected plane shape: {x.shape}")

        batch, slices, channels, height, width = x.shape
        x = x.contiguous().view(batch * slices, channels, height, width)

        feat_map = net(x)
        patch_tokens = feat_map.flatten(2).transpose(1, 2)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_pos = self._interpolate_pos(self.patch_pos_embed, patch_tokens.size(1))
        patch_tokens = patch_tokens + patch_pos.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        patch_tokens = self.patch_encoder(patch_tokens)
        patch_tokens = self.patch_norm(patch_tokens)

        slice_features = patch_tokens.mean(dim=1).view(batch, slices, self.vit_dim)
        slice_pos = self._interpolate_pos(self.slice_pos_embed, slices)
        slice_features = slice_features + slice_pos.to(device=slice_features.device, dtype=slice_features.dtype)

        cls = self.cls_token.to(device=slice_features.device, dtype=slice_features.dtype).expand(batch, -1, -1)
        slice_tokens = torch.cat([cls, slice_features], dim=1)
        slice_tokens = self.slice_encoder(slice_tokens)
        slice_tokens = self.slice_norm(slice_tokens)
        return self._pool_slice_tokens(slice_tokens)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("Input must be a list/tuple with axial, coronal and sagittal tensors.")

        planes = torch.stack(
            [
                self._encode_plane(self.axial, x[0]),
                self._encode_plane(self.coronal, x[1]),
                self._encode_plane(self.sagittal, x[2]),
            ],
            dim=1,
        )
        plane_embed = self.plane_embed.to(device=planes.device, dtype=planes.dtype)
        planes = self.plane_encoder(planes + plane_embed)
        return self.classifier(planes.flatten(1))


class EfficientNetB0ViT(EfficientNetB0_ViT):
    """Alias used by train_demo.py and older experiments."""

    pass
