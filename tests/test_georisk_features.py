"""CPU-only unit tests for utils.georisk_features.

Run with: pytest tests/test_georisk_features.py -v
"""
import math
import pytest
import torch

from utils.georisk_features import (
    GeoRiskConfig,
    masked_mean,
    masked_zscore,
)


def test_masked_mean_ignores_masked_positions():
    x = torch.tensor([[1.0, 2.0, 100.0, 4.0]])
    mask = torch.tensor([[True, True, False, True]])
    out = masked_mean(x, mask, dim=-1)
    # mean over [1, 2, 4] = 7/3
    assert torch.allclose(out, torch.tensor([7.0 / 3.0]), atol=1e-6)


def test_masked_mean_empty_row_returns_zero_not_nan():
    x = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[False, False]])
    out = masked_mean(x, mask, dim=-1)
    assert torch.isfinite(out).all(), "must not produce NaN/Inf on all-masked rows"


def test_masked_zscore_zero_variance_returns_zeros():
    x = torch.full((1, 5), 3.14)
    mask = torch.ones((1, 5), dtype=torch.bool)
    z = masked_zscore(x, mask, dim=-1)
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)


def test_masked_zscore_matches_hand_computation():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 999.0]])
    mask = torch.tensor([[True, True, True, True, False]])
    # mean = 2.5, var = 1.25, std = sqrt(1.25)
    z = masked_zscore(x, mask, dim=-1)
    std = math.sqrt(1.25)
    expected = torch.tensor([[(1 - 2.5) / std, (2 - 2.5) / std, (3 - 2.5) / std, (4 - 2.5) / std, 0.0]])
    assert torch.allclose(z, expected, atol=1e-5)


def test_masked_zscore_masked_positions_are_zero():
    x = torch.randn(2, 7)
    mask = torch.tensor([[True, True, False, True, True, False, True],
                        [True, False, True, False, True, True, True]])
    z = masked_zscore(x, mask, dim=-1)
    assert (z[~mask] == 0).all(), "masked positions must be zero in z-score output"


def test_georisk_config_defaults_match_spec():
    cfg = GeoRiskConfig()
    assert cfg.top_k == 64
    assert cfg.teacher_temperature == 1.0
    assert cfg.lambda_risk == 1.0
    assert cfg.lambda_relevance == 0.5
    assert cfg.lambda_alignment == 0.5
    assert cfg.lambda_kl == 1.0
    assert cfg.lambda_instability == 0.0
    assert cfg.lambda_unlearnability == 0.0
    assert cfg.weight_tau == 1.0
    assert cfg.weight_clip_min == 0.05
    assert cfg.weight_clip_max == 3.0
    assert cfg.stopgrad is True
