"""Building blocks for 3D spatiotemporal encoding."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """3D conv + norm + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder3D(nn.Module):
    """Lightweight 3D CNN encoder."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.stem = ConvBlock3D(in_channels, c)
        self.down1 = nn.Sequential(ConvBlock3D(c, c * 2, stride=2), ConvBlock3D(c * 2, c * 2))
        self.down2 = nn.Sequential(ConvBlock3D(c * 2, c * 4, stride=2), ConvBlock3D(c * 4, c * 4))
        self.out_channels = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return x


class TemporalConv(nn.Module):
    """Temporal mixing via 1D conv over encoded frames."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)"""
        return self.temporal(x)


class ConvLSTMCell3D(nn.Module):
    """Simplified ConvLSTM cell operating on 3D feature maps."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, d, h, w = x.shape
        if state is None:
            h_t = torch.zeros(b, self.hidden_dim, d, h, w, device=x.device, dtype=x.dtype)
            c_t = torch.zeros_like(h_t)
        else:
            h_t, c_t = state
        combined = torch.cat([x, h_t], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, (h_t, c_t)


class TemporalAttention(nn.Module):
    """Lightweight temporal self-attention over spatially pooled frame features."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        heads = max(1, min(int(num_heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)"""
        # (B, T, C)
        tokens = x.permute(0, 2, 1)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        out = self.norm(tokens + out)
        return out.permute(0, 2, 1)


class TemporalEncoder(nn.Module):
    """Encode per-frame 3D features with temporal module.

    ``temporal_mode``: ``temporal_conv`` | ``conv_lstm`` | ``temporal_attention``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        temporal_mode: str = "temporal_conv",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.frame_encoder = Encoder3D(in_channels, base_channels)
        feat_ch = self.frame_encoder.out_channels
        self.temporal_mode = temporal_mode
        hidden_dim = hidden_dim or feat_ch

        if temporal_mode == "conv_lstm":
            self.temporal = ConvLSTMCell3D(feat_ch, hidden_dim)
            self.out_channels = hidden_dim
        elif temporal_mode == "temporal_conv":
            self.temporal = TemporalConv(feat_ch)
            self.out_channels = feat_ch
        elif temporal_mode == "temporal_attention":
            self.temporal = TemporalAttention(feat_ch)
            self.out_channels = feat_ch
        else:
            raise ValueError(
                f"Unknown temporal_mode: {temporal_mode}. "
                "Use temporal_conv | conv_lstm | temporal_attention."
            )

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ):
        """
        Args:
            x: (B, C, T, D, H, W)
        Returns:
            aggregated (B, C_out, D', H', W'), or (sequence, aggregated) if
            ``return_sequence`` where sequence is (B, C_out, T, D', H', W').
        """
        b, c, t, d, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
        feats = self.frame_encoder(frames)
        _, fc, d2, h2, w2 = feats.shape
        stacked = feats.view(b, t, fc, d2, h2, w2).permute(0, 2, 1, 3, 4, 5).contiguous()

        if self.temporal_mode == "conv_lstm":
            state = None
            outputs = []
            for ti in range(t):
                h_t, state = self.temporal(stacked[:, :, ti], state)
                outputs.append(h_t)
            seq = torch.stack(outputs, dim=2)
            agg = outputs[-1]
        elif self.temporal_mode == "temporal_attention":
            pooled = stacked.mean(dim=(3, 4, 5))  # (B, C, T)
            mixed = self.temporal(pooled)
            weights = torch.softmax(mixed, dim=-1)
            temporal_ctx = (mixed * weights).sum(dim=-1)
            seq = stacked + temporal_ctx.view(b, fc, 1, 1, 1, 1)
            spatial = stacked.mean(dim=2)
            agg = spatial + temporal_ctx.view(b, fc, 1, 1, 1)
        else:
            pooled = stacked.mean(dim=(3, 4, 5))
            mixed = self.temporal(pooled)
            weights = torch.softmax(mixed, dim=-1)
            temporal_ctx = (mixed * weights).sum(dim=-1)
            seq = stacked + temporal_ctx.view(b, fc, 1, 1, 1, 1)
            spatial = stacked.mean(dim=2)
            agg = spatial + temporal_ctx.view(b, fc, 1, 1, 1)

        if return_sequence:
            return seq, agg
        return agg


class Decoder3D(nn.Module):
    """Upsampling decoder for reconstruction."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(in_channels // 2, in_channels // 4, 2, stride=2),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
