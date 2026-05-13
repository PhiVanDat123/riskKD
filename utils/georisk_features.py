"""GeoRiskKD per-token weight computation.

All functions are forward-only and intended to run under torch.no_grad() in production.
See docs/superpowers/specs/2026-05-13-georiskkd-design.md for the full design.
"""
from __future__ import annotations

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
