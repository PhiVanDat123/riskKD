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


from utils.georisk_features import project_reference_topk_with_labels


def test_project_topk_returns_k_plus_one_slots():
    B, T, V, K = 2, 3, 10, 4
    ref_logp = torch.randn(B, T, V).log_softmax(-1)
    labels = torch.zeros(B, T, dtype=torch.long)
    idx = project_reference_topk_with_labels(ref_logp, labels, top_k=K)
    assert idx.shape == (B, T, K + 1)
    assert idx.dtype == torch.long


def test_project_topk_label_in_last_slot():
    B, T, V, K = 1, 1, 6, 3
    ref_logp = torch.tensor([[[0.1, 0.5, 0.05, 0.2, 0.1, 0.05]]]).log()
    # top-3 of ref by logp = ids [1, 3, 0] (probs 0.5, 0.2, 0.1)
    labels = torch.tensor([[5]])  # not in top-3
    idx = project_reference_topk_with_labels(ref_logp, labels, top_k=K)
    assert idx[0, 0, -1].item() == 5, "label id must occupy the last slot"
    topk = set(idx[0, 0, :K].tolist())
    assert topk == {0, 1, 3}, f"first K slots must be ref top-K ids, got {topk}"


def test_project_topk_duplicate_when_label_already_in_topk():
    B, T, V, K = 1, 1, 6, 3
    ref_logp = torch.tensor([[[0.1, 0.5, 0.05, 0.2, 0.1, 0.05]]]).log()
    labels = torch.tensor([[1]])  # already top-1
    idx = project_reference_topk_with_labels(ref_logp, labels, top_k=K)
    # Per spec §7.4: the duplicate is intentional and acceptable for v1.
    assert idx[0, 0, -1].item() == 1
    assert (idx[0, 0] == 1).sum().item() == 2, "expected duplicate label slot"


def test_project_topk_clamps_to_vocab_minus_one():
    B, T, V = 1, 1, 5
    ref_logp = torch.randn(B, T, V).log_softmax(-1)
    labels = torch.zeros(B, T, dtype=torch.long)
    # request more than vocab supports; helper must clamp K to V-1
    idx = project_reference_topk_with_labels(ref_logp, labels, top_k=99)
    assert idx.shape == (B, T, V)  # (V-1) + 1 label slot = V
