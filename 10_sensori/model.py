import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Conv1DEncoder(nn.Module):
    """Encode channel-last sensor windows with a configurable 1D convolution stack."""

    def __init__(
        self,
        in_channels: int,
        norm_mode: str,
        conv_layer_config: Sequence[Sequence[int]],
        conv_bias: bool,
    ):
        super().__init__()
        if norm_mode not in {'group_norm', 'layer_norm'}:
            raise ValueError(f'Unsupported norm_mode={norm_mode!r}; expected "group_norm" or "layer_norm".')

        self.conv_layers = nn.ModuleList()
        for index, (out_channels, kernel_size, stride) in enumerate(conv_layer_config):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=conv_bias,
            )
            nn.init.kaiming_normal_(conv.weight)

            if norm_mode == 'layer_norm':
                norm = nn.LayerNorm(out_channels)
            elif index == 0:
                norm = nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                    affine=True,
                )
            else:
                norm = nn.Identity()

            # Keep these names and their registration order compatible with released weights.
            self.conv_layers.append(nn.ModuleDict({'layer_norm': norm, 'conv': conv}))
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected a 3D input (batch, time, channel), found {list(x.shape)}.')

        x = x.transpose(1, 2)  # B x C x T
        for layer in self.conv_layers:
            x = layer['conv'](x)
            norm = layer['layer_norm']
            if isinstance(norm, nn.LayerNorm):
                # LayerNorm operates on the last dimension, so expose channels there.
                x = norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = norm(x)
            x = F.gelu(x)

        return x.transpose(1, 2)  # B x T x C


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.positional_embedding = cfg.positional_embedding
        self.dropout = cfg.dropout
        encoder_dim = cfg.encoder_dim

        if self.positional_embedding == 'relative':
            self.pos_conv = nn.Conv1d(
                encoder_dim,
                encoder_dim,
                kernel_size=cfg.conv_pos,
                padding=cfg.conv_pos // 2,
                groups=cfg.conv_pos_groups,
            )
            std = math.sqrt(4 / (cfg.conv_pos * encoder_dim))
            nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
            nn.init.constant_(self.pos_conv.bias, 0)

            self.pos_conv = weight_norm(self.pos_conv, name='weight', dim=2)
        elif self.positional_embedding == 'absolute':
            max_len = 2000
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, encoder_dim, 2) * (-math.log(10000.0) / encoder_dim))
            positional_encoding = torch.zeros(max_len, 1, encoder_dim)
            positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
            positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_encoding', positional_encoding)
        else:
            raise ValueError(
                f'Unsupported positional_embedding={self.positional_embedding!r}; expected "relative" or "absolute".'
            )

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=encoder_dim,
                    nhead=cfg.encoder_attention_heads,
                    dim_feedforward=cfg.encoder_ffn_embed_dim,
                    dropout=self.dropout,
                    activation=cfg.activation_fn,
                    batch_first=True,
                    norm_first=True,
                    bias=True,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.layerdrop = cfg.encoder_layerdrop

    def forward(self, x):
        if self.positional_embedding == 'relative':
            x_conv = self.pos_conv(x.transpose(1, 2))[..., : x.size(1)]
            x_conv = F.gelu(x_conv)
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = x + self.positional_encoding[: x.size(1)].transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            # LayerDrop skips whole encoder layers during training only.
            if self.training and self.layerdrop > 0 and torch.rand(()) < self.layerdrop:
                continue
            x = layer(x)

        return self.layer_norm(x)


class Sensori(nn.Module):
    """Pretrained encoder for extracting sensor embeddings."""

    def __init__(self, cfg):
        super().__init__()

        self.window_len = int(cfg.window_len)
        if self.window_len < 1:
            raise ValueError(f'model.window_len must be positive, got {self.window_len}')

        encoder_dim = cfg.encoder_dim

        self.layer_norm = nn.LayerNorm(encoder_dim)

        self.feat_encoder = Conv1DEncoder(
            in_channels=cfg.in_channels,
            norm_mode=cfg.norm_mode,
            conv_layer_config=cfg.conv_layer_config,
            conv_bias=cfg.conv_bias,
        )

        self.post_feat_projection = nn.Linear(cfg.conv_layer_config[-1][0], encoder_dim)

        self.encoder = TransformerEncoder(cfg)

    def forward(self, x):
        B, T, W, C = x.shape
        features = self.feat_encoder(x.reshape(B * T, W, C))

        if features.size(1) != 1:
            raise ValueError(
                f'The convolutional encoder must return one vector per input window, got {features.size(1)}.'
            )

        features = features.squeeze(1).reshape(B, -1, self.window_len, features.size(-1)).mean(dim=2)
        features = self.layer_norm(self.post_feat_projection(features))
        return self.encoder(features).mean(dim=1)


def load_sensori(checkpoint_path, cfg, device='cpu'):
    """Load bare weights or a Lightning checkpoint for inference."""
    model = Sensori(cfg)

    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = weights.get('state_dict', weights)
        state_dict = {
            '.'.join(part for part in name.split('.') if part != '_orig_mod').removeprefix('model.'): tensor
            for name, tensor in state_dict.items()
        }
        # Pretraining heads are saved in the checkpoint but are not used for embeddings.
        state_dict = {
            name: tensor
            for name, tensor in state_dict.items()
            if name != 'mask_emb'
            and not name.startswith(('masked_projection.', 'rank_projection.', 'contrastive_projection.'))
        }
        model.load_state_dict(state_dict)

    return model.to(device).eval()
