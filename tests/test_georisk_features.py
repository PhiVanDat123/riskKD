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


from utils.georisk_features import compute_branch_local_alignment


def _logp_from_probs(p):
    return torch.log(torch.clamp(p, min=1e-12))


def test_alignment_high_when_ref_agrees_with_label_on_chosen_branch():
    # Single token; vocab=4; label=0; ref puts mass on label; student is uniform.
    B, T, V = 1, 1, 4
    p_ref = torch.tensor([[[0.85, 0.05, 0.05, 0.05]]])
    p_stu = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    labels = torch.tensor([[0]])
    branch_sign = torch.ones(B, T)
    A = compute_branch_local_alignment(
        _logp_from_probs(p_stu), _logp_from_probs(p_ref), labels, branch_sign, top_k=3,
    )
    # trust_dir ≈ +0.6 at idx 0, slightly negative elsewhere; pref_dir = +1 * (onehot - p_stu)
    # is +0.75 at idx 0, -0.25 elsewhere. Vectors point in the same direction → cosine ≈ +1.
    assert A.shape == (B, T)
    assert A.item() > 0.9


def test_alignment_flips_sign_on_rejected_branch():
    B, T, V = 1, 1, 4
    p_ref = torch.tensor([[[0.85, 0.05, 0.05, 0.05]]])
    p_stu = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    labels = torch.tensor([[0]])
    A_chosen = compute_branch_local_alignment(
        _logp_from_probs(p_stu), _logp_from_probs(p_ref), labels, torch.ones(B, T), top_k=3,
    )
    A_rejected = compute_branch_local_alignment(
        _logp_from_probs(p_stu), _logp_from_probs(p_ref), labels, -torch.ones(B, T), top_k=3,
    )
    assert torch.allclose(A_chosen, -A_rejected, atol=1e-5)


def test_alignment_is_finite_and_bounded():
    B, T, V = 3, 5, 32
    p_ref = torch.softmax(torch.randn(B, T, V), dim=-1)
    p_stu = torch.softmax(torch.randn(B, T, V), dim=-1)
    labels = torch.randint(0, V, (B, T))
    branch_sign = torch.cat([torch.ones(1, T), -torch.ones(1, T), torch.ones(1, T)], dim=0)
    A = compute_branch_local_alignment(
        _logp_from_probs(p_stu), _logp_from_probs(p_ref), labels, branch_sign, top_k=8,
    )
    assert torch.isfinite(A).all()
    assert (A.abs() <= 1.0 + 1e-5).all()


from utils.georisk_features import compute_relevance_salience, compute_instability_proxy


def test_relevance_is_abs_of_logps_margin_and_detached():
    margin = torch.tensor([[-1.5, 0.0, 2.0]], requires_grad=True)
    m = compute_relevance_salience(margin)
    assert torch.allclose(m, torch.tensor([[1.5, 0.0, 2.0]]))
    assert not m.requires_grad


def test_instability_is_abs_deviation_of_neg_logp():
    # per_token_logps = log p; NLL = -log p; deviation from masked mean.
    per_token_logps = torch.tensor([[-1.0, -2.0, -3.0, -999.0]])
    mask = torch.tensor([[True, True, True, False]])
    # N_t = [1, 2, 3, 999]; masked mean over first three = 2; |N - 2| = [1, 0, 1, ?]
    I = compute_instability_proxy(per_token_logps, mask)
    assert torch.allclose(I[0, :3], torch.tensor([1.0, 0.0, 1.0]), atol=1e-6)


def test_instability_handles_all_masked_row_without_nan():
    per_token_logps = torch.tensor([[-1.0, -2.0]])
    mask = torch.tensor([[False, False]])
    I = compute_instability_proxy(per_token_logps, mask)
    assert torch.isfinite(I).all()


from utils.georisk_features import softmax_with_budget


def test_softmax_with_budget_preserves_token_budget():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = softmax_with_budget(q, mask, tau=1.0, w_min=0.05, w_max=3.0)
    # Σ w over valid positions must equal |M_y| = 4
    assert torch.allclose((w * mask).sum(-1), mask.sum(-1).float(), atol=1e-5)


def test_softmax_with_budget_clamps_then_renormalizes():
    # Hugely skewed q: most mass on token 3, but w_max=2.0 should clamp it.
    q = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    w = softmax_with_budget(q, mask, tau=1.0, w_min=0.05, w_max=2.0)
    # All valid weights must lie in [w_min, w_max] AFTER renorm. Budget invariant must still hold.
    assert (w[mask] >= 0.05 - 1e-5).all()
    assert (w[mask] <= 2.0 + 1e-5).all()
    assert torch.allclose((w * mask).sum(-1), mask.sum(-1).float(), atol=1e-5)


def test_softmax_with_budget_masked_positions_are_zero():
    q = torch.randn(2, 6)
    mask = torch.tensor([[True, True, False, True, True, False],
                        [False, True, True, True, True, True]])
    w = softmax_with_budget(q, mask, tau=1.0, w_min=0.05, w_max=3.0)
    assert (w[~mask] == 0).all()


def test_softmax_with_budget_all_masked_row_does_not_nan():
    q = torch.randn(2, 4)
    mask = torch.tensor([[False, False, False, False],
                        [True, True, True, True]])
    w = softmax_with_budget(q, mask, tau=1.0, w_min=0.05, w_max=3.0)
    assert torch.isfinite(w).all()
    # All-masked row: every position must be 0 (mask zeroes it post-hoc).
    assert (w[0] == 0).all()
