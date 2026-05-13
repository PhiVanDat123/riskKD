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
