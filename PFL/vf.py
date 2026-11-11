"""Conditional vector field for rectified-flow / OT-CFM on a 1D trajectory.

Time embedding and encoder context are fused, then injected into 1D U-Net
residual blocks by scale-shift (AdaLN). Ego plan is not encoded again here;
it already sits inside the EPN context vector.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbed(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.SiLU(), nn.Linear(d * 2, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.d // 2
        f = torch.exp(-math.log(1e4) * torch.arange(h, device=t.device).float() / h)
        a = t.float().unsqueeze(1) * f.unsqueeze(0) * 1000.0
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1))


class CondEncoder(nn.Module):
    def __init__(self, cond_dim: int, ctx_dim: int, mode_dim: int = 0):
        super().__init__()
        self.t_emb = TimeEmbed(cond_dim)
        self.c_proj = nn.Linear(ctx_dim, 64)
        self.fuse = nn.Sequential(
            nn.Linear(cond_dim + 64, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.mode_proj = None
        if mode_dim:
            self.attach_mode(mode_dim, cond_dim)

    def attach_mode(self, mode_dim: int, cond_dim: int):
        self.mode_proj = nn.Linear(mode_dim, cond_dim)
        if self.mode_proj is not None:
            nn.init.zeros_(self.mode_proj.weight)
            nn.init.zeros_(self.mode_proj.bias)

    def forward(self, t: torch.Tensor, ctx: torch.Tensor,
                mode: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = self.fuse(torch.cat([self.t_emb(t), self.c_proj(ctx)], dim=-1))
        if self.mode_proj is not None:
            if mode is None or mode.shape != (ctx.shape[0], self.mode_proj.in_features):
                raise ValueError("mode must have shape [B, mode_dim]")
            cond = cond + self.mode_proj(mode)
        return cond


class RelationCrossAttention(nn.Module):
    """Horizon-aware relation adapter with an initially zero residual branch."""

    def __init__(self, channels: int, relation_dim: int, heads: int = 4,
                 max_length: int = 25):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.rel_proj = nn.Linear(relation_dim, channels)
        self.query_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.query_pos = nn.Parameter(torch.zeros(1, max_length, channels))
        self.relation_pos = nn.Parameter(torch.zeros(1, max_length, channels))
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.out_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _resize_position(position: torch.Tensor, length: int) -> torch.Tensor:
        if position.shape[1] == length:
            return position
        return F.interpolate(position.transpose(1, 2), size=length,
                             mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        q = x.transpose(1, 2)
        kv = self.rel_proj(relation)
        q = self.query_norm(q + self._resize_position(self.query_pos, q.shape[1]))
        kv = self.key_norm(kv + self._resize_position(self.relation_pos, kv.shape[1]))
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        return x + self.out_proj(attended).transpose(1, 2)


class ResBlock(nn.Module):
    """Dilated 1D residual block with AdaLN scale-shift conditioning.

    Dropout sits on the residual branch, after the modulation and before the
    skip add, which is where the DDPM-family blocks put it. nn.Dropout carries
    no parameters or buffers, so adding it leaves the state dict unchanged and
    checkpoints trained without it still load.
    """

    def __init__(self, ic: int, oc: int, cd: int, ks: int = 3, dil: int = 1, ng: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        p = dil * (ks - 1) // 2
        self.c1 = nn.Conv1d(ic, oc, ks, padding=p, dilation=dil)
        self.c2 = nn.Conv1d(oc, oc, ks, padding=p, dilation=dil)
        self.n1 = nn.GroupNorm(min(ng, oc), oc)
        self.n2 = nn.GroupNorm(min(ng, oc), oc)
        self.cp = nn.Linear(cd, oc * 2)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
        self.relation_adapter = None

    def attach_relation_adapter(self, relation_dim: int, heads: int = 4,
                                max_length: int = 25):
        self.relation_adapter = RelationCrossAttention(
            self.c2.out_channels, relation_dim, heads=heads, max_length=max_length)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                relation: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        s, sh = self.cp(c).unsqueeze(-1).chunk(2, dim=1)
        h = self.act(h * (1 + s) + sh)
        h = self.drop(h) + self.skip(x)
        if self.relation_adapter is not None:
            if relation is None:
                raise ValueError("relation is required by the cross-attention adapter")
            h = self.relation_adapter(h, relation)
        return h


class UNet1D(nn.Module):
    def __init__(self, td: int = 2, ch: Optional[List[int]] = None, cd: int = 128, ctxd: int = 112,
                 dropout: float = 0.0, relation_dim: int = 0, relation_channels: int = 16,
                 relation_cross_attention: bool = False, relation_heads: int = 4,
                 max_length: int = 25):
        super().__init__()
        ch = ch or [64, 128, 256]
        dil = [1, 2, 4]
        self.ctx2ch = nn.Linear(ctxd, 16)
        self.relation_dim = relation_dim
        self.relation_cross_attention = relation_cross_attention
        input_relation_dim = 0 if relation_cross_attention else relation_dim
        self.rel2ch = (nn.Sequential(nn.Linear(relation_dim, relation_channels), nn.SiLU(),
                                     nn.Linear(relation_channels, relation_channels))
                       if input_relation_dim else None)
        self.inp = nn.Conv1d(td + 16 + (relation_channels if input_relation_dim else 0), ch[0], 1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        self.skch: List[int] = []
        pc = ch[0]
        for i, c in enumerate(ch):
            d = dil[i] if i < len(dil) else 1
            self.enc.append(nn.ModuleList([ResBlock(pc, c, cd, 3, d, dropout=dropout),
                                           ResBlock(c, c, cd, 3, d, dropout=dropout)]))
            if i < len(ch) - 1:
                self.skch.append(c)
                self.down.append(nn.Conv1d(c, c, 2, stride=2))
            pc = c
        self.bot = nn.ModuleList([ResBlock(ch[-1], ch[-1], cd, dropout=dropout),
                                  ResBlock(ch[-1], ch[-1], cd, dropout=dropout)])
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        dc = ch[-1]
        for sc in reversed(self.skch):
            self.up.append(nn.ConvTranspose1d(dc, dc, 2, stride=2))
            self.dec.append(nn.ModuleList([ResBlock(dc + sc, sc, cd, dropout=dropout),
                                           ResBlock(sc, sc, cd, dropout=dropout)]))
            dc = sc
        self.out = nn.Sequential(nn.GroupNorm(min(8, dc), dc), nn.SiLU(), nn.Conv1d(dc, td, 1))
        # Attach adapters only after every baseline layer has been initialised. Thus the
        # shared weights match Full for the same seed, and zero output projections make
        # the complete vector field exactly Full-equivalent at step zero.
        if relation_cross_attention:
            if not relation_dim:
                raise ValueError("relation_cross_attention requires relation_dim")
            blocks = [b for level in self.enc for b in level] + list(self.bot)
            blocks += [b for level in self.dec for b in level]
            for block in blocks:
                block.attach_relation_adapter(relation_dim, relation_heads, max_length)

    def forward(self, xt: torch.Tensor, cond: torch.Tensor, ctx: torch.Tensor,
                relation: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = xt.shape
        cf = self.ctx2ch(ctx).unsqueeze(-1).expand(-1, -1, T)
        inputs = [xt.transpose(1, 2), cf]
        if self.relation_dim and not self.relation_cross_attention:
            if relation is None or relation.shape != (B, T, self.relation_dim):
                raise ValueError("relation must have shape [B, T, relation_dim]")
            inputs.append(self.rel2ch(relation).transpose(1, 2))
        h = self.inp(torch.cat(inputs, dim=1))
        sk = []
        for i, blks in enumerate(self.enc):
            for b in blks:
                h = b(h, cond, relation)
            if i < len(self.down):
                sk.append(h)
                h = self.down[i](h)
        for b in self.bot:
            h = b(h, cond, relation)
        for i, (u, blks) in enumerate(zip(self.up, self.dec)):
            h = u(h)
            s = sk[-(i + 1)]
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = torch.cat([h, s], dim=1)
            for b in blks:
                h = b(h, cond, relation)
        return self.out(h).transpose(1, 2)


class VectorField(nn.Module):
    def __init__(self, ctx_dim: int = 112, cond_dim: int = 128, traj_dim: int = 2,
                 channels: Optional[List[int]] = None, dropout: float = 0.0,
                 relation_dim: int = 0, relation_cross_attention: bool = False,
                 relation_heads: int = 4, mode_dim: int = 0, max_length: int = 25):
        super().__init__()
        self.ce = CondEncoder(cond_dim, ctx_dim)
        self.net = UNet1D(td=traj_dim, ch=channels or [64, 128, 256], cd=cond_dim, ctxd=ctx_dim,
                          dropout=dropout, relation_dim=relation_dim,
                          relation_cross_attention=relation_cross_attention,
                          relation_heads=relation_heads, max_length=max_length)
        # Defer this new module until the baseline vector-field layers exist so the
        # shared layers retain their same-seed initialisation.
        if mode_dim:
            self.ce.attach_mode(mode_dim, cond_dim)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor,
                relation: Optional[torch.Tensor] = None,
                mode: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(xt, self.ce(t, ctx, mode), ctx, relation)
