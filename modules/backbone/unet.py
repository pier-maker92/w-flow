import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import UNetBackboneConfig


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        half = embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )
        self.out_dim = embed_dim * 4

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, in_ch, out_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        num_heads = min(num_heads, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNetBackbone(nn.Module):
    """Time-conditioned U-Net velocity network for Phase 2 image experiments.

    Architecture: encoder (ResBlocks + downsample) → mid (ResBlock + Attn + ResBlock)
    → decoder (upsample + skip concat + ResBlocks). Attention applied at levels
    whose spatial resolution is in config.attention_resolutions.
    """

    def __init__(self, config: UNetBackboneConfig, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.last_bottleneck: Optional[torch.Tensor] = None
        self.last_quant_output = None
        self._attn_res = set(config.attention_resolutions)

        base = config.base_channels
        mults = config.channel_multipliers
        n_res = config.num_res_blocks
        t_dim: int

        self.time_embed = TimestepEmbedding(config.time_embed_dim)
        t_dim = self.time_embed.out_dim

        # x0-cond doubles input channels: [x_t ‖ x_0] on channel dim
        in_ch_first = in_channels * 2 if config.use_x0_cond else in_channels
        self.in_conv = nn.Conv2d(in_ch_first, base, 3, padding=1)

        # ------------------------------------------------------------------
        # Encoder
        # enc_chs[i] = output channels of encoder level i
        # ------------------------------------------------------------------
        enc_chs: list[int] = []
        ch = base
        self.enc_blocks: nn.ModuleList = nn.ModuleList()   # one per level
        self.enc_attns: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        for i, mult in enumerate(mults):
            out_ch = base * mult
            rbs = nn.ModuleList()
            for j in range(n_res):
                rbs.append(ResBlock(ch if j == 0 else out_ch, out_ch, t_dim))
            self.enc_blocks.append(rbs)
            self.enc_attns.append(AttentionBlock(out_ch))
            if i < len(mults) - 1:
                self.downsamples.append(Downsample(out_ch))
            else:
                self.downsamples.append(nn.Identity())
            enc_chs.append(out_ch)
            ch = out_ch

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.mid_res1 = ResBlock(ch, ch, t_dim)
        self.mid_attn = AttentionBlock(ch)
        self.mid_res2 = ResBlock(ch, ch, t_dim)

        # ------------------------------------------------------------------
        # Decoder — mirrors encoder levels in reverse order
        # dec level j mirrors enc level (N-1-j)
        # skip_ch = enc_chs[N-1-j]
        # ------------------------------------------------------------------
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.dec_attns: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()

        for j in range(len(mults) - 1, -1, -1):
            skip_ch = enc_chs[j]
            out_ch = base * mults[j]
            rbs = nn.ModuleList()
            for k in range(n_res):
                in_ch = (ch + skip_ch) if k == 0 else out_ch
                rbs.append(ResBlock(in_ch, out_ch, t_dim))
            self.dec_blocks.append(rbs)
            self.dec_attns.append(AttentionBlock(out_ch))
            if j > 0:
                self.upsamples.append(Upsample(out_ch))
            else:
                self.upsamples.append(nn.Identity())
            ch = out_ch

        groups = min(8, ch)
        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

        # Optional bottleneck VQ: GAP → VQ (1 global code/image) → broadcast
        self.vq: Optional[nn.Module] = None
        if config.use_bottleneck_vq:
            from ..quantizer.vq import VectorQuantizer
            from ..configs import VQConfig
            btn_ch = config.base_channels * config.channel_multipliers[-1]
            vq_cfg = VQConfig(
                num_embeddings=config.bottleneck_vq_num_embeddings,
                embedding_dim=btn_ch,
                commitment_weight=0.25,
                use_ema=config.bottleneck_vq_use_ema,
                embedding_init_scale=config.bottleneck_vq_init_scale,
            )
            self.vq = VectorQuantizer(vq_cfg)

    def _encode(self, x: torch.Tensor, t_emb: torch.Tensor):
        """Returns (bottleneck, skips). x must already have x_cond concatenated."""
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for enc_rbs, enc_attn, down in zip(self.enc_blocks, self.enc_attns, self.downsamples):
            for rb in enc_rbs:
                h = rb(h, t_emb)
            if h.shape[-1] in self._attn_res:
                h = enc_attn(h)
            skips.append(h)
            h = down(h)
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)
        return h, skips

    def _decode(self, h: torch.Tensor, skips: list, t_emb: torch.Tensor) -> torch.Tensor:
        for dec_rbs, dec_attn, up, skip in zip(
            self.dec_blocks, self.dec_attns, self.upsamples, reversed(skips)
        ):
            for k, rb in enumerate(dec_rbs):
                if k == 0:
                    h = torch.cat([h, skip], dim=1)
                h = rb(h, t_emb)
            if h.shape[-1] in self._attn_res:
                h = dec_attn(h)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                x_cond: Optional[torch.Tensor] = None,
                force_vq_index: Optional[int] = None) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        h, skips = self._encode(x, t_emb)
        self.last_bottleneck = h

        if self.vq is not None:
            B, C, H, W = h.shape
            if force_vq_index is not None:
                # Bypass VQ lookup — use specified codebook entry for all samples
                idx = torch.full((B,), force_vq_index, dtype=torch.long, device=h.device)
                z_q = self.vq.embedding(idx)          # [B, C]
                self.last_quant_output = None
            else:
                z_pool = h.mean(dim=[2, 3])           # [B, C] — global average pool
                quant_out = self.vq(z_pool)           # VQ on [B, C]
                z_q = quant_out.z_q                   # [B, C]
                self.last_quant_output = quant_out
            h = z_q[:, :, None, None].expand(B, C, H, W).contiguous()
        else:
            self.last_quant_output = None

        return self._decode(h, skips, t_emb)
