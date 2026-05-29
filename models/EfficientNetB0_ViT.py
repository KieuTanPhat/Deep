import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _build_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except Exception:
        return models.efficientnet_b0(pretrained=True)


class EfficientNetB0ViT(nn.Module):
    """EfficientNet-B0 slice encoder plus ViT-style cross-plane transformer.

    The baseline collapses all slices with max pooling before classification. This
    hybrid keeps every slice as a token, adds slice and plane embeddings, then
    lets a transformer learn ordering and cross-plane interactions.
    """

    def __init__(
        self,
        feature_dim=1280,
        vit_dim=384,
        vit_depth=2,
        vit_heads=6,
        vit_mlp_ratio=2.0,
        vit_dropout=0.2,
        classifier_dropout=0.35,
        max_slices=64,
        pooling="cls_attention",
    ):
        super().__init__()

        self.axial = _build_efficientnet_b0().features
        self.coronal = _build_efficientnet_b0().features
        self.sagittal = _build_efficientnet_b0().features

        self.feature_dim = int(feature_dim)
        self.vit_dim = int(vit_dim)
        vit_heads = int(vit_heads)
        if self.vit_dim % vit_heads != 0:
            raise ValueError("vit_dim must be divisible by vit_heads.")
        self.max_slices = max(1, int(max_slices))
        self.pooling = pooling
        valid_pooling = {"cls", "mean", "max", "attention", "cls_attention"}
        if self.pooling not in valid_pooling:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.feature_dim, self.vit_dim)
        self.input_norm = nn.LayerNorm(self.vit_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_dim))
        self.slice_pos_embed = nn.Parameter(torch.zeros(1, self.max_slices, self.vit_dim))
        self.plane_embed = nn.Parameter(torch.zeros(1, 3, self.vit_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim,
            nhead=vit_heads,
            dim_feedforward=int(self.vit_dim * float(vit_mlp_ratio)),
            dropout=float(vit_dropout),
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(vit_depth))

        self.attn_pool = nn.Sequential(
            nn.LayerNorm(self.vit_dim),
            nn.Linear(self.vit_dim, 1),
        )

        if self.pooling == "cls_attention":
            classifier_dim = self.vit_dim * 2
        else:
            classifier_dim = self.vit_dim

        self.head = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(classifier_dim, self.vit_dim),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(self.vit_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        init_fn = getattr(nn.init, "trunc_normal_", nn.init.normal_)
        init_fn(self.cls_token, std=0.02)
        init_fn(self.slice_pos_embed, std=0.02)
        init_fn(self.plane_embed, std=0.02)

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

    def _slice_position(self, slices, device, dtype):
        if slices <= self.max_slices:
            return self.slice_pos_embed[:, :slices, :].to(device=device, dtype=dtype)

        pos = self.slice_pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=slices, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def _extract_slice_features(self, net, x):
        # x can be [S, C, H, W] or [B, S, C, H, W].
        if x.dim() == 4:
            feat = net(x)
            feat = self.cnn_pool(feat).view(feat.size(0), -1)
            return feat.unsqueeze(0)

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        batch, slices, channels, height, width = x.shape
        x = x.contiguous().view(batch * slices, channels, height, width)
        feat = net(x)
        feat = self.cnn_pool(feat).view(feat.size(0), -1)
        return feat.view(batch, slices, -1)

    def _pool_encoded_tokens(self, encoded):
        cls_feat = encoded[:, 0, :]
        slice_tokens = encoded[:, 1:, :]

        if self.pooling == "cls":
            return cls_feat
        if self.pooling == "mean":
            return slice_tokens.mean(dim=1)
        if self.pooling == "max":
            return slice_tokens.max(dim=1)[0]

        scores = self.attn_pool(slice_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_feat = torch.sum(slice_tokens * weights, dim=1)

        if self.pooling == "attention":
            return attn_feat
        return torch.cat([cls_feat, attn_feat], dim=1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("Input must be a list/tuple with axial, coronal and sagittal tensors.")

        nets = (self.axial, self.coronal, self.sagittal)
        token_groups = []

        for plane_idx, (net, plane) in enumerate(zip(nets, x)):
            features = self._extract_slice_features(net, plane)
            tokens = self.proj(features)
            pos = self._slice_position(tokens.size(1), tokens.device, tokens.dtype)
            plane_embed = self.plane_embed[:, plane_idx:plane_idx + 1, :].to(
                device=tokens.device,
                dtype=tokens.dtype,
            )
            token_groups.append(tokens + pos + plane_embed)

        tokens = torch.cat(token_groups, dim=1)
        cls_tokens = self.cls_token.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.input_norm(tokens)

        # TransformerEncoder in older PyTorch versions expects [sequence, batch, dim].
        encoded = self.encoder(tokens.transpose(0, 1)).transpose(0, 1)
        fused = self._pool_encoded_tokens(encoded)
        return self.head(fused)
