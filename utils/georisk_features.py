"""GeoRiskKD per-token weight computation.

All functions are forward-only and intended to run under torch.no_grad() in production.
See docs/superpowers/specs/2026-05-13-georiskkd-design.md for the full design.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch


@dataclass
class GeoRiskConfig:
    """Hyperparameters for GeoRiskKD token-weight allocation.

    Built once per training step in radpo_concatenated_forward via `from_args(self.args)`
    and passed as a single kwarg into _radpo_get_batch_logps. Keeping these out of
    _radpo_get_batch_logps' kwarg list keeps the helper testable without the full
    trainer args namespace.
    """

    top_k: int = 64
    teacher_temperature: float = 1.0
    lambda_risk: float = 1.0
    lambda_relevance: float = 0.5
    lambda_alignment: float = 0.5
    lambda_kl: float = 1.0
    lambda_instability: float = 0.0
    lambda_unlearnability: float = 0.0
    weight_tau: float = 1.0
    weight_clip_min: float = 0.05
    weight_clip_max: float = 3.0
    stopgrad: bool = True

    @classmethod
    def from_args(cls, args) -> "GeoRiskConfig":
        """Build a GeoRiskConfig from a CustomDPOConfig-style args namespace."""
        return cls(
            top_k=int(getattr(args, "radpo_georisk_top_k", cls.top_k)),
            teacher_temperature=float(getattr(args, "radpo_georisk_teacher_temperature", cls.teacher_temperature)),
            lambda_risk=float(getattr(args, "radpo_georisk_lambda_risk", cls.lambda_risk)),
            lambda_relevance=float(getattr(args, "radpo_georisk_lambda_relevance", cls.lambda_relevance)),
            lambda_alignment=float(getattr(args, "radpo_georisk_lambda_alignment", cls.lambda_alignment)),
            lambda_kl=float(getattr(args, "radpo_georisk_lambda_kl", cls.lambda_kl)),
            lambda_instability=float(getattr(args, "radpo_georisk_lambda_instability", cls.lambda_instability)),
            lambda_unlearnability=float(getattr(args, "radpo_georisk_lambda_unlearnability", cls.lambda_unlearnability)),
            weight_tau=float(getattr(args, "radpo_georisk_weight_tau", cls.weight_tau)),
            weight_clip_min=float(getattr(args, "radpo_georisk_weight_clip_min", cls.weight_clip_min)),
            weight_clip_max=float(getattr(args, "radpo_georisk_weight_clip_max", cls.weight_clip_max)),
            stopgrad=bool(getattr(args, "radpo_georisk_stopgrad", cls.stopgrad)),
        )


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = -1,
                keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """Mean of x over `dim`, counting only mask=True positions. Returns 0 on all-masked rows."""
    mask_f = mask.to(x.dtype)
    num = (x * mask_f).sum(dim=dim, keepdim=keepdim)
    den = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def masked_zscore(x: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Per-row z-score over `dim` using only mask=True positions.

    On zero-variance rows or all-masked rows, returns 0 for every element of that row.
    Masked positions in the output are always 0.
    """
    mu = masked_mean(x, mask, dim=dim, keepdim=True)
    var = masked_mean((x - mu) ** 2, mask, dim=dim, keepdim=True)
    z = (x - mu) / var.clamp_min(eps).sqrt()
    return z * mask.to(z.dtype)


def project_reference_topk_with_labels(
    reference_distribution_logps: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,                        # [B, T]   long, already-zeroed at masked positions
    top_k: int,
) -> torch.Tensor:                               # [B, T, K+1]
    """Per-position vocabulary projection: top-K of reference logps with the observed label appended.

    Always allocates K+1 slots:
      - first K = reference top-K vocab ids (descending by reference logp);
      - last slot = the observed label id.

    If the label is already in the top-K, the last slot becomes a duplicate of one of the
    earlier slots; this is acceptable for v1 (it only rescales one component of the local
    cosine in compute_branch_local_alignment). See spec §7.4.

    `top_k` is clamped to `V - 1` so the appended label always has its own slot conceptually.
    """
    V = reference_distribution_logps.shape[-1]
    K = max(1, min(int(top_k), V - 1))
    topk_idx = reference_distribution_logps.topk(K, dim=-1).indices  # [B, T, K]
    label_idx = labels.unsqueeze(-1).to(topk_idx.dtype)              # [B, T, 1]
    return torch.cat([topk_idx, label_idx], dim=-1)                  # [B, T, K+1]


def compute_branch_local_alignment(
    student_distribution_logps: torch.Tensor,    # [B, T, V]
    reference_distribution_logps: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,                        # [B, T]
    branch_sign: torch.Tensor,                   # [B, T]  in {+1, -1}
    top_k: int,
    eps: float = 1e-8,
) -> torch.Tensor:                               # [B, T]
    """Branch-local logit-gradient proxy A_t = cosine(trust_dir, pref_dir).

    On the K+1-slot vocabulary {top_k(ref_t) ∪ {label_t}}:
      trust_dir = p_ref_k - p_student_k        # where the reference wants probability to move
      pref_dir  = branch_sign * (onehot_k - p_student_k)
                                                # +1: chosen → push toward label
                                                # -1: rejected → push away from label
      A_t       = cosine(trust_dir, pref_dir)

    A high A_t means the reference correction direction agrees with the branch-local
    preference update direction.
    """
    idx = project_reference_topk_with_labels(reference_distribution_logps, labels, top_k)  # [B,T,K+1]
    p_ref_k = reference_distribution_logps.gather(-1, idx).exp()                            # [B,T,K+1]
    p_stu_k = student_distribution_logps.gather(-1, idx).exp()                              # [B,T,K+1]
    onehot_k = (idx == labels.unsqueeze(-1)).to(p_ref_k.dtype)                              # [B,T,K+1]
    trust_dir = p_ref_k - p_stu_k
    pref_dir = branch_sign.unsqueeze(-1) * (onehot_k - p_stu_k)
    cos = torch.nn.functional.cosine_similarity(trust_dir, pref_dir, dim=-1, eps=eps)
    return cos.clamp(-1.0, 1.0)


def compute_relevance_salience(logps_margin: torch.Tensor) -> torch.Tensor:
    """m_t = |logps_margin|. v1 proxy for preference relevance (per-branch, observed-token).

    See spec §7.3. A real teacher-value change ψ is deferred to v2.
    """
    return logps_margin.detach().abs()


def compute_instability_proxy(per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """I_t = |N_t - masked_mean(N_t)| where N_t = -per_token_logps.

    Batch/sequence-relative outlier-loss proxy, NOT true sharpness. See spec §7.5;
    do not call this 'sharpness' in logs or paper claims.
    """
    N = -per_token_logps.detach()
    return (N - masked_mean(N, loss_mask, dim=-1, keepdim=True)).abs()


def softmax_with_budget(
    q: torch.Tensor,              # [B, T]
    loss_mask: torch.Tensor,      # [B, T] bool
    tau: float,
    w_min: float,
    w_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:                # [B, T]
    """Fixed-budget softmax allocation: w_t = |M_y| · softmax(q/τ), then clamp, then renormalize.

    Invariant on rows with at least one valid token:
        (w * loss_mask).sum(-1) == loss_mask.sum(-1)

    All-masked rows return all-zeros (the caller multiplies the weights by the mask anyway, so
    this is safe). Masked positions in the output are always 0.
    """
    valid = loss_mask.to(q.dtype).sum(-1, keepdim=True)        # [B, 1]
    has_valid = (valid > 0)                                     # [B, 1]
    # Pre-softmax: -inf on masked positions, but for all-masked rows replace q with 0 to dodge softmax(-inf,...).
    q_masked = q.masked_fill(~loss_mask, float("-inf"))
    safe_q = torch.where(has_valid.expand_as(q_masked), q_masked, torch.zeros_like(q_masked))
    w = torch.softmax(safe_q / max(tau, eps), dim=-1) * valid.clamp_min(1.0)   # [B, T]
    # Two-phase projected-budget allocation.
    #
    # Phase 1 (upper projection): iteratively cap tokens exceeding w_max, redistributing the
    # excess budget proportionally among the remaining tokens.  Proportional redistribution
    # preserves the softmax score ordering and is numerically safe because only tokens with
    # positive weight can exceed w_max.
    #
    # Phase 2 (lower projection): after all over-max tokens are fixed, floor remaining tokens
    # at w_min and renormalize back to budget.  We iterate because renormalization after
    # flooring can push previously-fine tokens over w_max again (rare but possible).
    mask_f = loss_mask.to(w.dtype)
    budget = valid.clamp_min(1.0)  # [B, 1]

    # ── Phase 1: cap-and-redistribute (over w_max) ──────────────────────────────────────────
    fixed_hi = torch.zeros_like(loss_mask)   # mask of tokens fixed at w_max
    over = torch.zeros_like(loss_mask)       # initialise so else-clause can check it
    for _iter in range(64):
        over = (w > w_max + eps) & loss_mask & ~fixed_hi
        if not over.any():
            break
        # Fix newly-over-max tokens at w_max.
        fixed_hi = fixed_hi | over
        w = torch.where(over, torch.full_like(w, w_max), w)
        # Redistribute the residual budget to non-fixed tokens proportionally.
        free = loss_mask & ~fixed_hi
        fixed_sum = (w * fixed_hi.to(w.dtype)).sum(-1, keepdim=True)          # [B, 1]
        remaining = (budget - fixed_sum).clamp_min(0.0)                        # [B, 1]
        free_w = w * free.to(w.dtype)
        free_sum = free_w.sum(-1, keepdim=True).clamp_min(eps)
        w = torch.where(free, free_w * remaining / free_sum, w)
    else:
        if over.any():
            warnings.warn(
                f"softmax_with_budget Phase 1 did not converge in 64 iterations; "
                f"{int(over.sum().item())} tokens still violate w_max",
                RuntimeWarning,
            )

    # ── Phase 2: floor-and-renorm (under w_min) ─────────────────────────────────────────────
    # After phase 1 every valid token is ≤ w_max.  Now floor at w_min and renorm.
    # Renorm can only push values upward, so w_min invariant is preserved.
    # If renorm pushes a token over w_max, run another phase-1 pass.
    for _ in range(64):
        w = w.clamp(min=w_min) * mask_f
        w_sum = (w * mask_f).sum(-1, keepdim=True).clamp_min(eps)
        w = w * budget / w_sum * mask_f
        # Check for new over-max violations introduced by the renorm.
        over = (w > w_max + eps) & loss_mask
        if not over.any():
            break
        # Fix new over-max tokens and redistribute (same logic as phase 1).
        fixed_hi2 = over
        w = torch.where(over, torch.full_like(w, w_max), w)
        free2 = loss_mask & ~fixed_hi2
        fixed_sum2 = (w * fixed_hi2.to(w.dtype)).sum(-1, keepdim=True)
        remaining2 = (budget - fixed_sum2).clamp_min(0.0)
        free_w2 = w * free2.to(w.dtype)
        free_sum2 = free_w2.sum(-1, keepdim=True).clamp_min(eps)
        w = torch.where(free2, free_w2 * remaining2 / free_sum2, w)

    # Zero out masked positions and all-masked rows.
    w = torch.where(loss_mask, w, torch.zeros_like(w))
    w = torch.where(has_valid.expand_as(w), w, torch.zeros_like(w))
    return w


def _shannon_entropy(weights: torch.Tensor, mask_f: torch.Tensor, valid_count: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of weights normalized to a probability distribution over valid tokens."""
    p = (weights * mask_f) / valid_count.clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum()


def compute_georisk_token_weights(
    *,
    student_logits: torch.Tensor,                  # [B, T, V] (post-slice; pre-temperature)
    reference_logits: torch.Tensor,                # [B, T, V]
    distribution_logps: torch.Tensor,              # [B, T, V] = student log-softmax
    reference_distribution_logps: torch.Tensor,    # [B, T, V] = reference log-softmax
    labels: torch.Tensor,                          # [B, T]    (caller has already replaced -100 with 0)
    loss_mask: torch.Tensor,                       # [B, T] bool
    branch_sign: torch.Tensor,                     # [B, T]    +1 chosen rows, -1 rejected rows
    per_position_kl: torch.Tensor,                 # [B, T]
    per_position_risk_ratio: torch.Tensor,         # [B, T]
    per_token_logps: torch.Tensor,                 # [B, T]
    per_reference_token_logps: torch.Tensor,       # [B, T]  (kept for symmetry; unused in v1)
    logps_margin: torch.Tensor,                    # [B, T]
    config: GeoRiskConfig,
):
    """Compute fixed-budget per-token GeoRiskKD weights.

    Returns:
        weights:       [B, T]  (stop-gradient if config.stopgrad)
        georisk_stats: dict[str, torch.Tensor scalar] — diagnostics for logging.
                       Assumes B is even and the first B/2 rows are chosen, second half rejected.
                       See spec §11.
    """
    del per_reference_token_logps  # not used in v1; documented for symmetry with spec
    device = student_logits.device
    dtype = student_logits.dtype
    B, T = loss_mask.shape
    assert B % 2 == 0, (
        f"compute_georisk_token_weights expects an even batch dim with first B/2 chosen "
        f"and second B/2 rejected (got B={B})."
    )

    with torch.no_grad():
        # --- features ---
        r_t = per_position_risk_ratio.detach()
        D_t = per_position_kl.detach()

        if config.lambda_relevance != 0.0:
            m_t = compute_relevance_salience(logps_margin)
        else:
            m_t = torch.zeros((B, T), device=device, dtype=dtype)

        if config.lambda_instability != 0.0:
            I_t = compute_instability_proxy(per_token_logps, loss_mask)
        else:
            I_t = torch.zeros((B, T), device=device, dtype=dtype)

        if config.lambda_unlearnability != 0.0:
            N_t = -per_token_logps.detach()
        else:
            N_t = torch.zeros((B, T), device=device, dtype=dtype)

        if config.lambda_alignment != 0.0:
            A_t = compute_branch_local_alignment(
                distribution_logps,
                reference_distribution_logps,
                labels,
                branch_sign,
                top_k=config.top_k,
            )
        else:
            A_t = torch.zeros((B, T), device=device, dtype=dtype)

        # --- nonfinite scrub ---
        nonfinite = torch.zeros((), device=device)
        cleaned = []
        for f in (r_t, m_t, A_t, D_t, I_t, N_t):
            finite = torch.isfinite(f)
            nonfinite = nonfinite + (~finite).to(nonfinite.dtype).sum()
            cleaned.append(torch.where(finite, f, torch.zeros_like(f)))
        r_t, m_t, A_t, D_t, I_t, N_t = cleaned

        # --- z-score per sequence ---
        z_r = masked_zscore(r_t, loss_mask) if config.lambda_risk != 0.0 else r_t
        z_m = masked_zscore(m_t, loss_mask) if config.lambda_relevance != 0.0 else m_t
        z_A = masked_zscore(A_t, loss_mask) if config.lambda_alignment != 0.0 else A_t
        z_D = masked_zscore(D_t, loss_mask) if config.lambda_kl != 0.0 else D_t
        z_I = masked_zscore(I_t, loss_mask) if config.lambda_instability != 0.0 else I_t
        z_N = masked_zscore(N_t, loss_mask) if config.lambda_unlearnability != 0.0 else N_t

        q = (
            config.lambda_risk * z_r
            + config.lambda_relevance * z_m
            + config.lambda_alignment * z_A
            - config.lambda_kl * z_D
            - config.lambda_instability * z_I
            - config.lambda_unlearnability * z_N
        )

        weights = softmax_with_budget(
            q, loss_mask,
            tau=config.weight_tau,
            w_min=config.weight_clip_min,
            w_max=config.weight_clip_max,
        )

        # --- diagnostics ---
        mask_f = loss_mask.to(weights.dtype)
        valid_count = mask_f.sum().clamp_min(1)

        def _mean_over_mask(t):
            return (t * mask_f).sum() / valid_count

        def _branch_mean(t):
            half = t.shape[0] // 2
            denom_c = mask_f[:half].sum().clamp_min(1)
            denom_r = mask_f[half:].sum().clamp_min(1)
            return (t[:half] * mask_f[:half]).sum() / denom_c, (t[half:] * mask_f[half:]).sum() / denom_r

        # top-10% weighted positions for the *_top_weighted stats
        flat_w = (weights * mask_f).flatten()
        n_top = max(1, int(valid_count.item() * 0.1))
        top_idx = flat_w.topk(min(n_top, flat_w.numel())).indices

        stats = {
            "georisk/r_mean": _mean_over_mask(r_t),
            "georisk/m_mean": _mean_over_mask(m_t),
            "georisk/A_mean": _mean_over_mask(A_t),
            "georisk/D_mean": _mean_over_mask(D_t),
            "georisk/I_mean": _mean_over_mask(I_t),
            "georisk/N_mean": _mean_over_mask(N_t),
            "georisk/nonfinite_count": nonfinite,
            "georisk/effective_tokens": (weights * mask_f).sum(),
            "georisk/top10_weight_mass": flat_w.index_select(0, top_idx).sum() / valid_count,
            "georisk/token_weight_entropy": _shannon_entropy(weights, mask_f, valid_count),
        }
        cm, rm = _branch_mean(weights)
        stats["georisk/weight_chosen_mean"] = cm
        stats["georisk/weight_rejected_mean"] = rm
        for name, f in (("r", r_t), ("m", m_t), ("A", A_t), ("D", D_t), ("I", I_t), ("N", N_t)):
            stats[f"georisk/{name}_top_weighted"] = f.flatten().index_select(0, top_idx).mean()

    if config.stopgrad:
        weights = weights.detach()
    return weights, stats
