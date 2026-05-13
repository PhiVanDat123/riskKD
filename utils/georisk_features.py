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
