"""PFL encoder: temporal self-attention + ego-conditioned social pooling.

History is encoded with MultiheadAttention over time (not LSTM). Neighbours
are still scattered onto the PiP/EPN 25x5 grid so the ego plan can enter as a
cell, then pooled by the original social conv. When use_attention is on, the
target also cross-attends to neighbour tokens and the ego-plan token.

On top of that grid path the ego plan also enters as a *relational* feature:
physical relative quantities between the ego's planned pose and the target's
own constant-velocity extrapolation, encoded by a GRU and added to the target's
dynamics embedding (see RelationalEgoEncoder).
"""
import torch
from torch import nn


def cv_velocity(hist, gate=None):
    """Per-step displacement of the target, averaged over the last few steps.

    targsHist is expressed relative to the target's own current position, so
    hist[-1] is exactly zero; a finite difference on the last frame alone would
    see little but NGSIM's position noise.

    `gate` masks axes out of the extrapolation. It defaults to longitudinal only,
    because NGSIM's lateral coordinate is noisy enough that a finite-difference
    lateral velocity is mostly measurement noise: extrapolating it 25 steps takes
    the residual std at 5 s from 1.58 ft up to 2.77 ft, i.e. the baseline makes the
    lateral channel harder rather than easier. Longitudinally it is the opposite --
    64.3 ft of std and a 76 ft mean collapse to 21.0 ft of std and no mean.
    """
    h = hist.permute(1, 0, 2)                       # [B, T, 2]
    k = min(3, h.shape[1] - 1)
    if k <= 0:
        v0 = torch.zeros_like(h[:, -1])
    else:
        v0 = (h[:, -1] - h[:, -1 - k]) / float(k)   # [B, 2]
    return v0 if gate is None else v0 * gate


def cv_baseline(hist, out_length, gate=None):
    """Constant-velocity extrapolation of the target, [B, out_length, 2]."""
    v0 = cv_velocity(hist, gate)
    steps = torch.arange(1, out_length + 1, device=hist.device, dtype=v0.dtype)
    return v0.unsqueeze(1) * steps.view(1, -1, 1)


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=64, n_heads=4, dropout=0.1, max_len=50):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pool_q = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: [T, B, C] -> pooled [B, embed_dim]"""
        x = x.permute(1, 0, 2)
        b, t, _ = x.shape
        e = self.embed(x) + self.pos[:, :t, :]
        out, _ = self.attn(e, e, e)
        e = self.norm1(e + out)
        e = self.norm2(e + self.ffn(e))
        q = self.pool_q.expand(b, -1, -1)
        p, _ = self.pool_attn(q, e, e)
        return self.pool_norm(p).squeeze(1)


class RelationalEgoEncoder(nn.Module):
    """Ego plan injected as a per-agent relational feature.

    At each horizon step the ego's planned pose is compared against the target's
    own constant-velocity extrapolation to the same step, giving

        r_t = [ds_t, dd_t, dv_par_t, dv_perp_t, dpsi_t]

    -- longitudinal / lateral gap, their per-step rates, and the heading of the
    relative motion in the target's frame. These depend only on physical relative
    quantities, never on an agent index, so they leave the permutation structure
    of the surrounding encoder untouched. The sequence is read by a GRU and the
    result is added to the target's dynamics embedding, following
    Z~ = Z + MLP(GRU(r)).

    The output projection is zero-initialised so the module starts as the identity
    and only earns its contribution during training.
    """

    # ds, dd, dv_par, dv_perp, dpsi. Feet and feet-per-step; dpsi is already in radians.
    SCALE = (50.0, 12.0, 5.0, 1.0, 1.0)

    def __init__(self, hidden, out_dim):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(5, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, plan_fwd, cv, return_sequence=False, force_zero=False):
        """plan_fwd, cv: [B, T, 2] with channel 0 lateral, channel 1 longitudinal."""
        rel = plan_fwd - cv
        plan_d = plan_fwd[:, 1:] - plan_fwd[:, :-1]
        cv_d = cv[:, 1:] - cv[:, :-1]
        plan_v = torch.cat([plan_d[:, :1], plan_d], dim=1)
        cv_v = torch.cat([cv_d[:, :1], cv_d], dim=1)
        dv = plan_v - cv_v
        plan_heading = torch.atan2(plan_v[..., 0], plan_v[..., 1])
        cv_heading = torch.atan2(cv_v[..., 0], cv_v[..., 1])
        dpsi = torch.atan2(torch.sin(plan_heading - cv_heading),
                           torch.cos(plan_heading - cv_heading))
        r = torch.stack([rel[..., 1], rel[..., 0], dv[..., 1], dv[..., 0],
                         dpsi], dim=-1)
        r = r / r.new_tensor(self.SCALE)
        if force_zero:
            r = torch.zeros_like(r)
        _, hn = self.gru(self.inp(r))
        pooled = self.mlp(hn.squeeze(0))
        return (pooled, r) if return_sequence else pooled


class PFLEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_planning = args.use_planning
        self.use_attention = args.use_attention
        self.use_relational = getattr(args, "use_relational", True) and self.use_planning
        self.temporal_relational = bool(getattr(args, "temporal_relational", False))
        self.temporal_only = bool(getattr(args, "temporal_only", False))
        self.temporal_relation_zero = bool(getattr(args, "temporal_relation_zero", False))
        if self.temporal_relational and not self.use_relational:
            raise ValueError("temporal_relational requires planning and relational conditioning")
        if (self.temporal_only or self.temporal_relation_zero) and not self.temporal_relational:
            raise ValueError("temporal_only/temporal_relation_zero require temporal_relational")
        self.relation_control = getattr(args, "relation_control", "none")
        if self.use_relational and self.relation_control != "none":
            raise ValueError("relation_control is only valid when relational conditioning is off")
        self.grid_size = args.grid_size
        self.out_length = args.out_length
        self.encoder_size = args.encoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        self.order = args.order
        self.intent_condition = bool(getattr(args, "intent_condition", False))
        self.relational_gate = bool(getattr(args, "relational_gate", False))
        if self.relational_gate and not (self.intent_condition and self.use_relational):
            raise ValueError("relational_gate requires intent_condition and relational planning")
        if self.intent_condition:
            hd = self.social_context_size + self.dynamics_encoding_size
            self.intent_lat = nn.Linear(hd, args.num_lat_classes)
            self.intent_lon = nn.Linear(hd, args.num_lon_classes)
            self.history_pool = nn.MaxPool2d((2, 1), padding=(1, 0))
        if self.relational_gate:
            self.rel_gate = nn.Sequential(
                nn.Linear(hd + args.num_lat_classes + args.num_lon_classes + 4, 64),
                nn.GELU(), nn.Linear(64, self.dynamics_encoding_size))
            nn.init.zeros_(self.rel_gate[-1].weight)
            nn.init.zeros_(self.rel_gate[-1].bias)
        n_heads = int(getattr(args, "attn_heads", 4))
        dropout = float(getattr(args, "attn_dropout", 0.1))
        max_len = max(args.in_length, args.out_length) + 8

        self.motion_dim = 2 + (1 if self.order >= 1 else 0) + (1 if self.order >= 2 else 0) + (
            1 if self.order >= 3 else 0)
        self.hist_enc = TemporalAttentionEncoder(
            self.motion_dim, self.encoder_size, n_heads, dropout, max_len)
        self.plan_enc = TemporalAttentionEncoder(
            2, self.encoder_size, n_heads, dropout, max_len)
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            self.leaky_relu,
        )
        if self.use_planning:
            self.plan_conv_social = nn.Sequential(
                nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
                self.leaky_relu,
                nn.MaxPool2d((3, 3), stride=2),
                nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
                self.leaky_relu,
            )
            self.pool_after_merge = nn.MaxPool2d((2, 2), padding=(1, 0))
        else:
            self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))

        if self.use_relational:
            self.rel_ego = RelationalEgoEncoder(self.encoder_size, self.dynamics_encoding_size)
        elif self.relation_control == "target_cv":
            # Same trainable architecture and parameter count as the relational
            # encoder, but it sees a zero ego path. This isolates added sequence-model
            # capacity from information carried by the ego plan.
            self.rel_control = RelationalEgoEncoder(
                self.encoder_size, self.dynamics_encoding_size)
        gate = [1.0 if getattr(args, "cv_lateral", False) else 0.0, 1.0]
        self.register_buffer("cv_gate", torch.tensor(gate))

        if self.use_attention:
            self.social_attn = nn.MultiheadAttention(
                self.encoder_size, n_heads, dropout=dropout, batch_first=True)
            self.social_attn_norm = nn.LayerNorm(self.encoder_size)
            self.social_attn_proj = nn.Linear(self.encoder_size, self.dynamics_encoding_size)

    def _motion(self, xy, vel, acc, coll):
        parts = [xy]
        if self.order >= 1:
            parts.append(vel)
        if self.order >= 2:
            parts.append(acc)
        if self.order >= 3:
            parts.append(coll)
        return torch.cat(parts, dim=-1)

    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                lat_enc, lon_enc, vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist,
                coll_hist, dest=None):
        return self.soc_encode(
            nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
            lat_enc, lon_enc, vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist,
            coll_hist, dest)

    def soc_encode(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
                   lat_enc, lon_enc, vel_nbrs, acc_nbrs, coll_nbrs, vel_hist, acc_hist,
                   coll_hist, dest=None):
        dyn_raw = self.hist_enc(self._motion(targsHist, vel_hist, acc_hist, coll_hist))
        dyn_enc = self.leaky_relu(self.dyn_emb(dyn_raw))

        if nbsHist.shape[1] == 0:
            nbrs_enc = dyn_raw.new_zeros(0, self.encoder_size)
        else:
            nbrs_enc = self.hist_enc(self._motion(nbsHist, vel_nbrs, acc_nbrs, coll_nbrs))

        nbrs_grid = torch.zeros_like(nbsMask).float()
        if nbrs_enc.numel():
            nbrs_grid = nbrs_grid.masked_scatter_(nbsMask, nbrs_enc)
        nbrs_grid = nbrs_grid.view(nbrs_grid.shape[0], nbrs_grid.shape[1] * nbrs_grid.shape[2],
                                   nbrs_grid.shape[3])
        nbrs_grid = nbrs_grid.view(nbrs_grid.shape[0], 5, 25, nbrs_grid.shape[2])
        nbrs_grid = nbrs_grid.permute(0, 3, 2, 1)
        nbrs_feat = self.nbrs_conv_social(nbrs_grid)

        intent = None
        if self.intent_condition:
            # No plan, ground-truth maneuver or target future enters this branch.
            history_dyn = dyn_enc
            if self.use_attention:
                hist_tokens = nbrs_grid.permute(0, 2, 3, 1).reshape(
                    dyn_raw.shape[0], -1, self.encoder_size)
                hist_pad = ~nbsMask[..., 0].permute(0, 2, 1).reshape(dyn_raw.shape[0], -1)
                hist_pad = hist_pad & ~hist_pad.all(dim=1, keepdim=True)
                hist_attn, _ = self.social_attn(dyn_raw.unsqueeze(1), hist_tokens,
                                               hist_tokens, key_padding_mask=hist_pad)
                history_dyn = history_dyn + self.social_attn_proj(
                    self.social_attn_norm(hist_attn.squeeze(1)))
            history_ctx = torch.cat((self.history_pool(nbrs_feat).flatten(1), history_dyn), 1)
            intent = torch.cat((self.intent_lat(history_ctx).softmax(-1),
                                self.intent_lon(history_ctx).softmax(-1)), 1)

        plan_raw = None
        if self.use_planning:
            plan_raw = self.plan_enc(planFut)
            plan_grid = torch.zeros_like(planMask).float()
            plan_grid = plan_grid.masked_scatter_(planMask, plan_raw)
            plan_grid = plan_grid.view(plan_grid.shape[0], plan_grid.shape[1] * plan_grid.shape[2],
                                       plan_grid.shape[3])
            plan_grid = plan_grid.view(plan_grid.shape[0], 5, 25, plan_grid.shape[2])
            plan_grid = plan_grid.permute(0, 3, 2, 1)
            plan_feat = self.plan_conv_social(plan_grid)
            social_context = self.pool_after_merge(torch.cat((nbrs_feat, plan_feat), dim=3))
        else:
            social_context = self.pool_after_merge(nbrs_feat)

        social = social_context.view(-1, self.social_context_size)

        if self.use_relational:
            # planFut arrives time-reversed from getPlanFuture; flip it back so step k
            # of the plan lines up with step k of the target's own extrapolation.
            plan_fwd = planFut.permute(1, 0, 2).flip(1)
            cv = cv_baseline(targsHist, self.out_length, self.cv_gate)
            relation_out = self.rel_ego(
                plan_fwd, cv, return_sequence=self.temporal_relational,
                force_zero=self.temporal_relation_zero)
            if self.temporal_relational:
                relational, relation_seq = relation_out
            else:
                relational, relation_seq = relation_out, None
            if self.relational_gate:
                # Ego cell identifies a currently observed neighbour, not a future pose.
                # Relative position and its recent per-step change use history only.
                k = min(3, nbsHist.shape[0] - 1)
                motion = torch.cat((nbsHist[-1],
                                    (nbsHist[-1] - nbsHist[-1-k]) / max(k, 1)), -1)
                geom_grid = dyn_raw.new_zeros(*nbsMask.shape[:3], 4)
                geom_grid = geom_grid.masked_scatter_(nbsMask[..., :1].expand_as(geom_grid), motion)
                geometry = (geom_grid * planMask[..., :1]).sum((1, 2))
                geometry = torch.tanh(geometry / geometry.new_tensor([12., 50., 1., 5.]))
                gate = 2.0 * torch.sigmoid(self.rel_gate(torch.cat(
                    (history_ctx, intent, geometry), 1)))
                relational = relational * gate  # identity at initialisation
            if not self.temporal_only:
                dyn_enc = dyn_enc + relational
        elif self.relation_control == "target_cv":
            cv = cv_baseline(targsHist, self.out_length, self.cv_gate)
            dyn_enc = dyn_enc + self.rel_control(torch.zeros_like(cv), cv)

        if self.use_attention:
            b = dyn_raw.shape[0]
            # nbrs_grid is [B, C, lon, lat], so its token order is lon*5 + lat, while
            # nbsMask is [B, lat, lon, C] and flattens as lat*25 + lon. The two agree
            # only where lon == 6*lat; without this transpose the padding mask hides
            # the real neighbours and lets the attention read empty cells.
            tokens = nbrs_grid.permute(0, 2, 3, 1).reshape(b, -1, self.encoder_size)
            pad = ~nbsMask[..., 0].permute(0, 2, 1).reshape(b, -1)
            if plan_raw is not None:
                tokens = torch.cat([tokens, plan_raw.unsqueeze(1)], dim=1)
                pad = torch.cat([pad, torch.zeros(b, 1, dtype=torch.bool, device=pad.device)], dim=1)
            pad = pad & pad.all(dim=1, keepdim=True).logical_not()
            attn_out, _ = self.social_attn(dyn_raw.unsqueeze(1), tokens, tokens,
                                           key_padding_mask=pad)
            dyn_enc = dyn_enc + self.social_attn_proj(self.social_attn_norm(attn_out.squeeze(1)))

        ctx = torch.cat((social, dyn_enc), 1)
        ctx = torch.cat((ctx, intent), 1) if intent is not None else ctx
        return (ctx, relation_seq) if self.temporal_relational else ctx
