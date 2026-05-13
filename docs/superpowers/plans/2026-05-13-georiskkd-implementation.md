# GeoRiskKD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `radpo_token_weight_mode="georisk"` mode to the existing Ra-DPO / RiskKD trainer that computes a stop-gradient, fixed-budget per-token allocation from six branch-local forward-only features.

**Architecture:** New helper module `utils/georisk_features.py` (pure functions + dataclass), plumbed into the existing `_radpo_get_batch_logps` integration point via a `GeoRiskConfig` dataclass built once in `radpo_concatenated_forward`. Existing kl_inv path is preserved; `radpo_loss_fn` is unchanged. Three new Qwen3 recipes (baseline + two GeoRiskKD variants).

**Tech Stack:** PyTorch 2.8 / transformers 4.51 / trl 0.12 / DeepSpeed Zero-3 (no new dependencies). CPU-only unit tests via `pytest`.

**Reference spec:** [`docs/superpowers/specs/2026-05-13-georiskkd-design.md`](../specs/2026-05-13-georiskkd-design.md)
**Codebase entry point:** [`scripts/run_distill_dpo.py`](../../../scripts/run_distill_dpo.py)

---

## Conventions used in this plan

- Every step that writes code shows the **full** code block — no "similar to Task N" references.
- Commit messages are concrete; engineer should copy them verbatim.
- Tests are CPU-only and live under `tests/`. The repo has no existing test infra; tests are runnable via the project venv with `pytest`.
- All new tensor ops are under `torch.no_grad()` in production; tests verify the stop-gradient invariant explicitly.
- Each task ends with a commit. Do not batch commits across tasks.

---

## Task 1: Scaffold `utils/georisk_features.py` with `GeoRiskConfig`, `masked_mean`, `masked_zscore`

**Files:**
- Create: `utils/georisk_features.py`
- Create: `tests/__init__.py`
- Create: `tests/test_georisk_features.py`

- [ ] **Step 1.1: Write the failing tests for `masked_mean` and `masked_zscore`**

Create `tests/__init__.py` as an empty file. Create `tests/test_georisk_features.py`:

```python
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
```

- [ ] **Step 1.2: Run the tests — verify they fail with ImportError**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.georisk_features'` (or `ImportError` on the names).

- [ ] **Step 1.3: Create `utils/georisk_features.py` with the dataclass and the two functions**

```python
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
```

- [ ] **Step 1.4: Run the tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 6 passed.

- [ ] **Step 1.5: Commit**

```bash
git add utils/georisk_features.py tests/__init__.py tests/test_georisk_features.py
git commit -m "feat(georisk): scaffold GeoRiskConfig + masked_mean / masked_zscore

First slice of utils/georisk_features.py: the dataclass that carries
GeoRiskKD hyperparameters from CustomDPOConfig into _radpo_get_batch_logps
and two masked reductions used by every downstream feature.

CPU-only unit tests in tests/test_georisk_features.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `project_reference_topk_with_labels`

**Files:**
- Modify: `utils/georisk_features.py` (append)
- Modify: `tests/test_georisk_features.py` (append)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_georisk_features.py`:

```python
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
```

- [ ] **Step 2.2: Run the new tests — verify they fail**

```bash
pytest tests/test_georisk_features.py::test_project_topk_returns_k_plus_one_slots -v
```

Expected: `ImportError: cannot import name 'project_reference_topk_with_labels'`.

- [ ] **Step 2.3: Implement the function**

Append to `utils/georisk_features.py`:

```python
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
```

- [ ] **Step 2.4: Run tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 10 passed total.

- [ ] **Step 2.5: Commit**

```bash
git add utils/georisk_features.py tests/test_georisk_features.py
git commit -m "feat(georisk): project_reference_topk_with_labels (K+1 slot, dup ok)

Per-position vocabulary projection used by the alignment feature. Always
allocates K+1 slots and tolerates the duplicate when the label is already
in the reference top-K, per spec §7.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `compute_branch_local_alignment`

**Files:**
- Modify: `utils/georisk_features.py` (append)
- Modify: `tests/test_georisk_features.py` (append)

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_georisk_features.py`:

```python
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
```

- [ ] **Step 3.2: Run the new tests — verify they fail**

```bash
pytest tests/test_georisk_features.py::test_alignment_high_when_ref_agrees_with_label_on_chosen_branch -v
```

Expected: ImportError on `compute_branch_local_alignment`.

- [ ] **Step 3.3: Implement the function**

Append to `utils/georisk_features.py`:

```python
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
```

- [ ] **Step 3.4: Run tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 13 passed total.

- [ ] **Step 3.5: Commit**

```bash
git add utils/georisk_features.py tests/test_georisk_features.py
git commit -m "feat(georisk): branch-local logit-gradient alignment A_t

cosine(ref-student trust dir, branch_sign * (onehot - student) pref dir)
on the K+1-slot top-K projection. Forward-only; no extra teacher pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `compute_relevance_salience` and `compute_instability_proxy`

**Files:**
- Modify: `utils/georisk_features.py` (append)
- Modify: `tests/test_georisk_features.py` (append)

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/test_georisk_features.py`:

```python
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
```

- [ ] **Step 4.2: Run new tests — verify they fail**

```bash
pytest tests/test_georisk_features.py::test_relevance_is_abs_of_logps_margin_and_detached -v
```

Expected: ImportError.

- [ ] **Step 4.3: Implement the two small features**

Append to `utils/georisk_features.py`:

```python
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
```

- [ ] **Step 4.4: Run tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 16 passed total.

- [ ] **Step 4.5: Commit**

```bash
git add utils/georisk_features.py tests/test_georisk_features.py
git commit -m "feat(georisk): relevance m_t and instability I_t proxies

m_t = |logps_margin|; I_t = |-logp - masked_mean(-logp)|. Both forward-only,
both detached. Per spec §7.3 / §7.5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `softmax_with_budget`

**Files:**
- Modify: `utils/georisk_features.py` (append)
- Modify: `tests/test_georisk_features.py` (append)

- [ ] **Step 5.1: Write the failing tests**

Append to `tests/test_georisk_features.py`:

```python
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
```

- [ ] **Step 5.2: Run new tests — verify they fail**

```bash
pytest tests/test_georisk_features.py::test_softmax_with_budget_preserves_token_budget -v
```

Expected: ImportError.

- [ ] **Step 5.3: Implement `softmax_with_budget`**

Append to `utils/georisk_features.py`:

```python
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
    # Clamp and renormalize back to budget.
    w = w.clamp(w_min, w_max)
    w_sum = (w * loss_mask.to(w.dtype)).sum(-1, keepdim=True).clamp_min(eps)
    w = w * valid.clamp_min(1.0) / w_sum
    # Zero out masked positions and all-masked rows.
    w = torch.where(loss_mask, w, torch.zeros_like(w))
    w = torch.where(has_valid.expand_as(w), w, torch.zeros_like(w))
    return w
```

- [ ] **Step 5.4: Run tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 20 passed total.

- [ ] **Step 5.5: Commit**

```bash
git add utils/georisk_features.py tests/test_georisk_features.py
git commit -m "feat(georisk): fixed-budget softmax allocation with clamp + renorm

w_t = |M_y| * softmax(q/τ), clamp(w_min,w_max), renormalize. Budget
invariant holds after clipping. All-masked rows return zeros.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `compute_georisk_token_weights` orchestrator

**Files:**
- Modify: `utils/georisk_features.py` (append)
- Modify: `tests/test_georisk_features.py` (append)

- [ ] **Step 6.1: Write the failing tests**

Append to `tests/test_georisk_features.py`:

```python
from utils.georisk_features import compute_georisk_token_weights


def _make_dummy_batch(B=4, T=6, V=12, K=4, seed=0):
    """Build a synthetic batch shaped like the post-slice tensors inside _radpo_get_batch_logps.

    The first B/2 rows are 'chosen', the second B/2 are 'rejected'.
    """
    torch.manual_seed(seed)
    student_logits = torch.randn(B, T, V)
    reference_logits = torch.randn(B, T, V)
    distribution_logps = student_logits.log_softmax(-1)
    reference_distribution_logps = reference_logits.log_softmax(-1)
    labels = torch.randint(0, V, (B, T))
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss_mask[:, -1] = False  # last position is masked
    branch_sign = torch.cat([torch.ones(B // 2, T), -torch.ones(B // 2, T)], dim=0)
    per_position_kl = torch.randn(B, T).abs()
    per_position_risk_ratio = torch.randn(B, T)
    per_token_logps = distribution_logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    per_reference_token_logps = reference_distribution_logps.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    logps_margin = per_token_logps - per_reference_token_logps
    return dict(
        student_logits=student_logits,
        reference_logits=reference_logits,
        distribution_logps=distribution_logps,
        reference_distribution_logps=reference_distribution_logps,
        labels=labels,
        loss_mask=loss_mask,
        branch_sign=branch_sign,
        per_position_kl=per_position_kl,
        per_position_risk_ratio=per_position_risk_ratio,
        per_token_logps=per_token_logps,
        per_reference_token_logps=per_reference_token_logps,
        logps_margin=logps_margin,
    )


def test_orchestrator_returns_budget_preserving_weights():
    cfg = GeoRiskConfig(top_k=4, lambda_instability=1.0, lambda_unlearnability=0.0)
    batch = _make_dummy_batch()
    w, stats = compute_georisk_token_weights(config=cfg, **batch)
    mask = batch["loss_mask"]
    # Σ w over valid positions == |M_y|
    assert torch.allclose((w * mask).sum(-1), mask.sum(-1).float(), atol=1e-4)
    # Stats dict populated with expected keys
    expected = {
        "georisk/r_mean", "georisk/m_mean", "georisk/A_mean",
        "georisk/D_mean", "georisk/I_mean", "georisk/N_mean",
        "georisk/r_top_weighted", "georisk/A_top_weighted",
        "georisk/token_weight_entropy", "georisk/effective_tokens",
        "georisk/top10_weight_mass", "georisk/weight_chosen_mean",
        "georisk/weight_rejected_mean", "georisk/nonfinite_count",
    }
    assert expected.issubset(stats.keys()), f"missing: {expected - stats.keys()}"


def test_orchestrator_lambda_alignment_zero_short_circuits():
    """When lambda_alignment=0, compute_branch_local_alignment must not be called."""
    import utils.georisk_features as gf
    cfg = GeoRiskConfig(top_k=4, lambda_alignment=0.0)
    batch = _make_dummy_batch()

    call_count = {"n": 0}
    original = gf.compute_branch_local_alignment

    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    gf.compute_branch_local_alignment = _spy
    try:
        compute_georisk_token_weights(config=cfg, **batch)
    finally:
        gf.compute_branch_local_alignment = original
    assert call_count["n"] == 0, "lambda_alignment=0 must short-circuit the alignment computation"


def test_orchestrator_stopgrad_returns_detached_weights():
    cfg = GeoRiskConfig(top_k=4, stopgrad=True)
    batch = _make_dummy_batch()
    # Make student logits require grad so naive code would propagate.
    batch["student_logits"].requires_grad_(True)
    batch["distribution_logps"] = batch["student_logits"].log_softmax(-1)
    w, _ = compute_georisk_token_weights(config=cfg, **batch)
    assert not w.requires_grad


def test_orchestrator_all_masked_row_produces_no_nans():
    cfg = GeoRiskConfig(top_k=4)
    batch = _make_dummy_batch()
    batch["loss_mask"][0] = False  # row 0 fully masked
    w, stats = compute_georisk_token_weights(config=cfg, **batch)
    assert torch.isfinite(w).all()
    for k, v in stats.items():
        assert torch.isfinite(v).all(), f"stat {k} is non-finite"


def test_orchestrator_nonfinite_feature_replaced_with_zero():
    cfg = GeoRiskConfig(top_k=4)
    batch = _make_dummy_batch()
    batch["per_position_risk_ratio"][0, 0] = float("nan")
    w, stats = compute_georisk_token_weights(config=cfg, **batch)
    assert torch.isfinite(w).all()
    assert stats["georisk/nonfinite_count"].item() >= 1
```

- [ ] **Step 6.2: Run new tests — verify they fail**

```bash
pytest tests/test_georisk_features.py::test_orchestrator_returns_budget_preserving_weights -v
```

Expected: ImportError on `compute_georisk_token_weights`.

- [ ] **Step 6.3: Implement the orchestrator**

Append to `utils/georisk_features.py`:

```python
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


def _shannon_entropy(weights: torch.Tensor, mask_f: torch.Tensor, valid_count: torch.Tensor,
                     eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of weights normalized to a probability distribution over valid tokens."""
    p = (weights * mask_f) / valid_count.clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum()
```

- [ ] **Step 6.4: Run tests — verify they pass**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 25 passed total.

- [ ] **Step 6.5: Commit**

```bash
git add utils/georisk_features.py tests/test_georisk_features.py
git commit -m "feat(georisk): compute_georisk_token_weights orchestrator

Top-level helper that wires all six features + z-score + fixed-budget
softmax into a single call. λ=0 short-circuits feature computation;
non-finite features are replaced with zero and counted; stop-gradient
honored. Returns a diagnostics dict with branch-split means and
top-10% weighted feature stats.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Extend `CustomDPOConfig` with `radpo_georisk_*` fields

**Files:**
- Modify: `scripts/run_distill_dpo.py:200-204` (radpo_token_weight_* group)
- Modify: `tests/test_georisk_features.py` (add `from_args` integration test)

- [ ] **Step 7.1: Locate the existing radpo_token_weight_* fields**

Open `scripts/run_distill_dpo.py` and find the block beginning at line 200:

```python
radpo_token_weight_mode: Optional[str] = field(default="none", metadata={"help": "Token-level weighting inside Ra-DPO: 'none' or 'kl_inv' (w_t = exp(-alpha * per_position_KL_t))."})
```

This is where we insert the new fields and update the mode enum docstring.

- [ ] **Step 7.2: Write a failing test for `GeoRiskConfig.from_args` reading the new field set**

Append to `tests/test_georisk_features.py`:

```python
def test_geo_risk_config_from_args_reads_radpo_georisk_fields():
    class FakeArgs:
        radpo_georisk_top_k = 32
        radpo_georisk_teacher_temperature = 2.5
        radpo_georisk_lambda_risk = 0.7
        radpo_georisk_lambda_relevance = 0.0
        radpo_georisk_lambda_alignment = 1.5
        radpo_georisk_lambda_kl = 0.9
        radpo_georisk_lambda_instability = 0.2
        radpo_georisk_lambda_unlearnability = 0.1
        radpo_georisk_weight_tau = 0.8
        radpo_georisk_weight_clip_min = 0.01
        radpo_georisk_weight_clip_max = 5.0
        radpo_georisk_stopgrad = False

    cfg = GeoRiskConfig.from_args(FakeArgs())
    assert cfg.top_k == 32
    assert cfg.teacher_temperature == 2.5
    assert cfg.lambda_risk == 0.7
    assert cfg.lambda_relevance == 0.0
    assert cfg.lambda_alignment == 1.5
    assert cfg.lambda_kl == 0.9
    assert cfg.lambda_instability == 0.2
    assert cfg.lambda_unlearnability == 0.1
    assert cfg.weight_tau == 0.8
    assert cfg.weight_clip_min == 0.01
    assert cfg.weight_clip_max == 5.0
    assert cfg.stopgrad is False
```

- [ ] **Step 7.3: Run — this test will already pass (from_args was implemented in Task 1)**

```bash
pytest tests/test_georisk_features.py::test_geo_risk_config_from_args_reads_radpo_georisk_fields -v
```

Expected: PASS. (We kept `from_args` in Task 1; this just locks the field name contract.)

- [ ] **Step 7.4: Update the `radpo_token_weight_mode` help string and insert georisk fields**

Edit `scripts/run_distill_dpo.py`. Find the line at ~200:

```python
    radpo_token_weight_mode: Optional[str] = field(default="none", metadata={"help": "Token-level weighting inside Ra-DPO: 'none' or 'kl_inv' (w_t = exp(-alpha * per_position_KL_t))."})
```

Replace with:

```python
    radpo_token_weight_mode: Optional[str] = field(default="none", metadata={"help": "Token-level weighting inside Ra-DPO: 'none', 'kl_inv' (w_t = exp(-alpha * per_position_KL_t)), or 'georisk' (fixed-budget softmax over branch-local features; see utils/georisk_features.py)."})
```

Then, immediately after the existing `radpo_token_weight_target` field declaration (the line ending `"all" or "risk" (risk correction only)."})`), insert the 12 new fields:

```python
    # --- GeoRiskKD (radpo_token_weight_mode="georisk") ---
    radpo_georisk_top_k: Optional[int] = field(default=64, metadata={"help": "Per-position top-K reference vocabulary size for the alignment feature. K is clamped to V-1."})
    radpo_georisk_teacher_temperature: Optional[float] = field(default=1.0, metadata={"help": "Teacher softmax temperature for GeoRiskKD feature extraction. Keep at 1.0 to reuse reference_distribution_logps without an extra log-softmax pass."})
    radpo_georisk_lambda_risk: Optional[float] = field(default=1.0, metadata={"help": "Score weight on z-scored per-token risk salience r_t."})
    radpo_georisk_lambda_relevance: Optional[float] = field(default=0.5, metadata={"help": "Score weight on z-scored relevance proxy m_t = |logps_margin|."})
    radpo_georisk_lambda_alignment: Optional[float] = field(default=0.5, metadata={"help": "Score weight on z-scored branch-local logit-gradient alignment A_t."})
    radpo_georisk_lambda_kl: Optional[float] = field(default=1.0, metadata={"help": "Score weight (subtracted) on z-scored teacher-student KL D_t."})
    radpo_georisk_lambda_instability: Optional[float] = field(default=0.0, metadata={"help": "Score weight (subtracted) on z-scored instability proxy I_t. 0.0 by default."})
    radpo_georisk_lambda_unlearnability: Optional[float] = field(default=0.0, metadata={"help": "Score weight (subtracted) on z-scored unlearnability proxy N_t = -log pi_theta. 0.0 by default."})
    radpo_georisk_weight_tau: Optional[float] = field(default=1.0, metadata={"help": "Softmax temperature for the fixed-budget allocation."})
    radpo_georisk_weight_clip_min: Optional[float] = field(default=0.05, metadata={"help": "Lower clip on per-token weight after the softmax allocation (renormalized to budget afterwards)."})
    radpo_georisk_weight_clip_max: Optional[float] = field(default=3.0, metadata={"help": "Upper clip on per-token weight after the softmax allocation."})
    radpo_georisk_stopgrad: Optional[bool] = field(default=True, metadata={"help": "Detach GeoRiskKD weights before applying them to the Ra-DPO loss. Default True."})
```

- [ ] **Step 7.5: Verify the file still imports cleanly**

```bash
python -c "import ast; ast.parse(open('scripts/run_distill_dpo.py').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 7.6: Commit**

```bash
git add scripts/run_distill_dpo.py tests/test_georisk_features.py
git commit -m "feat(georisk): add radpo_georisk_* fields to CustomDPOConfig

Twelve new dataclass fields wire the GeoRiskKD hyperparameters from the
recipe YAMLs through to GeoRiskConfig.from_args(self.args). Extends the
radpo_token_weight_mode enum docstring to include 'georisk'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Wire `_radpo_get_batch_logps` — branch_sign, georisk_config, mutual exclusion, 8-tuple return

**Files:**
- Modify: `scripts/run_distill_dpo.py:553-642` (`_radpo_get_batch_logps`)

- [ ] **Step 8.1: Read the current function signature**

The function at line 553 currently has this signature:

```python
def _radpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor,
                           labels: torch.LongTensor, weights: torch.FloatTensor = None,
                           confidence_level: float = 0.5, is_split_risk_ratio: bool = True,
                           is_cal_risk_distribution_logps: bool = False,
                           average_log_prob: bool = False,
                           token_weight_mode: str = "none", token_weight_alpha: float = 0.0,
                           token_weight_normalize: bool = False,
                           token_weight_target: str = "all"):
```

We add two kwargs (`branch_sign`, `georisk_config`) at the end so call-site ordering is preserved.

- [ ] **Step 8.2: Add imports at the top of `scripts/run_distill_dpo.py`**

Find the existing `from utils.compress_logits import ...` line (or similar `utils` import). After it, add:

```python
from utils.georisk_features import GeoRiskConfig, compute_georisk_token_weights
```

If no such line exists, add the import near the other top-level imports in the file (after the `import torch` block).

- [ ] **Step 8.3: Update the function signature**

In `scripts/run_distill_dpo.py:553-560`, replace the signature with:

```python
def _radpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor,
                           labels: torch.LongTensor, weights: torch.FloatTensor = None,
                           confidence_level: float = 0.5, is_split_risk_ratio: bool = True,
                           is_cal_risk_distribution_logps: bool = False,
                           average_log_prob: bool = False,
                           token_weight_mode: str = "none", token_weight_alpha: float = 0.0,
                           token_weight_normalize: bool = False,
                           token_weight_target: str = "all",
                           branch_sign: torch.Tensor = None,
                           georisk_config: GeoRiskConfig = None):
```

- [ ] **Step 8.4: Replace the existing `if weights is None: ... else: ...` block with the new georisk-aware block**

Lines 594-602 currently read:

```python
    if weights is None:
        if token_weight_mode == "kl_inv" and token_weight_alpha > 0:
            # Downweight tokens where policy & reference disagree wildly (the cross-size noise).
            # per_position_kl is already [B, T-1] (computed on the sliced labels above).
            weights = torch.exp(-token_weight_alpha * per_position_kl.detach())
        else:
            weights = torch.ones_like(logps_margin)
    else:
        weights = weights[:, 1:].clone()
```

Replace with:

```python
    georisk_stats: dict = {}
    if token_weight_mode == "georisk" and weights is not None:
        raise ValueError(
            "radpo_token_weight_mode='georisk' is mutually exclusive with an explicit "
            "per-token `weights` tensor. Either disable WPO weighting (set use_weighting=False / "
            "remove chosen_weight/rejected_weight from the batch) or use mode='kl_inv'/'none'."
        )

    if weights is None:
        if token_weight_mode == "kl_inv" and token_weight_alpha > 0:
            # Downweight tokens where policy & reference disagree wildly (the cross-size noise).
            # per_position_kl is already [B, T-1] (computed on the sliced labels above).
            weights = torch.exp(-token_weight_alpha * per_position_kl.detach())
        elif token_weight_mode == "georisk":
            if georisk_config is None:
                raise ValueError("radpo_token_weight_mode='georisk' requires a GeoRiskConfig "
                                 "passed via `georisk_config=`; build it in radpo_concatenated_forward "
                                 "with GeoRiskConfig.from_args(self.args).")
            if branch_sign is None:
                raise ValueError("radpo_token_weight_mode='georisk' requires `branch_sign` "
                                 "(+1 for chosen rows, -1 for rejected) passed from "
                                 "radpo_concatenated_forward.")
            # branch_sign comes in unsliced from the caller; slice consistently with labels.
            branch_sign_sliced = branch_sign[:, 1:]
            weights, georisk_stats = compute_georisk_token_weights(
                student_logits=logits,
                reference_logits=reference_logits,
                distribution_logps=distribution_logps,
                reference_distribution_logps=reference_distribution_logps,
                labels=labels,
                loss_mask=loss_mask,
                branch_sign=branch_sign_sliced,
                per_position_kl=per_position_kl,
                per_position_risk_ratio=per_position_risk_ratio,
                per_token_logps=per_token_logps,
                per_reference_token_logps=per_reference_token_logps,
                logps_margin=logps_margin,
                config=georisk_config,
            )
        else:
            weights = torch.ones_like(logps_margin)
    else:
        weights = weights[:, 1:].clone()
```

- [ ] **Step 8.5: Extend both return tuples with `georisk_stats`**

Lines 622-642 contain the two `return (...)` blocks. Modify them to return 8 elements:

Replace the `if average_log_prob: ... else: ...` return block with:

```python
    if average_log_prob:
        denom = loss_mask.sum(-1).clamp_min(1)
        return (
            (logps_margin * margin_weights * loss_mask).sum(-1) / denom,
            weighted_position_kl / denom,
            raw_position_kl / denom,
            (per_position_risk_ratio * risk_weights * loss_mask).sum(-1) / denom,
            (per_token_logps * logp_weights * loss_mask).sum(-1) / denom,
            mean_token_weight,
            effective_token_count,
            georisk_stats,
        )
    else:
        return (
            (logps_margin * margin_weights * loss_mask).sum(-1),
            weighted_position_kl,
            raw_position_kl,
            (per_position_risk_ratio * risk_weights * loss_mask).sum(-1),
            (per_token_logps * logp_weights * loss_mask).sum(-1),
            mean_token_weight,
            effective_token_count,
            georisk_stats,
        )
```

- [ ] **Step 8.6: Verify file parses**

```bash
python -c "import ast; ast.parse(open('scripts/run_distill_dpo.py').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 8.7: Commit**

```bash
git add scripts/run_distill_dpo.py
git commit -m "feat(georisk): wire _radpo_get_batch_logps for mode=georisk

Adds branch_sign / georisk_config kwargs, raises ValueError on mutually
exclusive (mode=georisk, weights=tensor) configurations, and extends
the return contract to an 8-tuple (georisk_stats appended; empty dict
for non-georisk modes).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire `radpo_concatenated_forward` — build branch_sign, build GeoRiskConfig, unpack the 8-tuple

**Files:**
- Modify: `scripts/run_distill_dpo.py:1633-1728` (`radpo_concatenated_forward`)

- [ ] **Step 9.1: Insert `branch_sign` construction and `georisk_config` build**

Find the block at line 1693 starting with `(`, which begins the unpack of `_radpo_get_batch_logps`. Immediately *before* that block (just after the existing mu-anneal block, after line 1691 `self._current_radpo_mu = mu`), insert:

```python
        # GeoRiskKD: branch_sign labels each concatenated row as chosen (+1) or rejected (-1).
        # max_length comes from labels.shape[1] above (unsliced); _radpo_get_batch_logps will slice
        # it consistently with labels via branch_sign[:, 1:].
        branch_sign = torch.cat(
            [
                torch.ones(num_examples, max_length, device=labels.device, dtype=torch.float32),
                -torch.ones(num_examples, max_length, device=labels.device, dtype=torch.float32),
            ],
            dim=0,
        )
        georisk_config = (
            GeoRiskConfig.from_args(self.args)
            if getattr(self.args, "radpo_token_weight_mode", "none") == "georisk"
            else None
        )
```

- [ ] **Step 9.2: Pass `branch_sign` and `georisk_config` into the call**

Modify the call at line 1701:

```python
        (
            all_logps_margin,
            all_weighted_position_kl,
            all_raw_position_kl,
            all_position_risk_ratio,
            all_logps,
            all_mean_token_weight,
            all_effective_token_count,
            all_georisk_stats,
        ) = _radpo_get_batch_logps(
            all_logits, ref_all_logits, labels, concatenated_weights,
            confidence_level=mu,
            is_split_risk_ratio=self.args.is_split_risk_ratio,
            is_cal_risk_distribution_logps=self.args.is_cal_risk_distribution_logps,
            average_log_prob=False,
            token_weight_mode=getattr(self.args, "radpo_token_weight_mode", "none"),
            token_weight_alpha=getattr(self.args, "radpo_token_weight_alpha", 0.0),
            token_weight_normalize=getattr(self.args, "radpo_token_weight_normalize", False),
            token_weight_target=getattr(self.args, "radpo_token_weight_target", "all"),
            branch_sign=branch_sign,
            georisk_config=georisk_config,
        )
```

- [ ] **Step 9.3: Append `all_georisk_stats` to the function's return tuple**

Modify the `return (...)` block at lines 1713-1728 to add the stats dict as the final element. After the existing 14 elements add `all_georisk_stats`:

```python
        return (
            all_logps_margin[:num_examples],        # chosen_logps_margin
            all_logps_margin[num_examples:],        # rejected_logps_margin
            all_weighted_position_kl[:num_examples], # chosen_weighted_position_kl
            all_weighted_position_kl[num_examples:], # rejected_weighted_position_kl
            all_raw_position_kl[:num_examples],     # chosen_raw_position_kl
            all_raw_position_kl[num_examples:],     # rejected_raw_position_kl
            all_position_risk_ratio[:num_examples], # chosen_position_risk_ratio
            all_position_risk_ratio[num_examples:], # rejected_position_risk_ratio
            all_logps[:num_examples].detach(),      # chosen_logps
            all_logps[num_examples:].detach(),      # rejected_logps
            all_mean_token_weight[:num_examples],   # chosen_mean_token_weight
            all_mean_token_weight[num_examples:],   # rejected_mean_token_weight
            all_effective_token_count[:num_examples], # chosen_effective_token_count
            all_effective_token_count[num_examples:], # rejected_effective_token_count
            all_georisk_stats,                       # dict[str, scalar] (empty unless mode=georisk)
        )
```

- [ ] **Step 9.4: Update the function docstring**

Find the function's docstring at lines 1634-1643:

```python
        """Forward pass for Ra-DPO: computes log-prob margins, KL, and CVaR risk ratios.

        Returns:
            chosen_logps_margin, rejected_logps_margin,
            chosen_weighted_position_kl, rejected_weighted_position_kl,
            chosen_raw_position_kl, rejected_raw_position_kl,
            chosen_position_risk_ratio, rejected_position_risk_ratio,
            chosen_logps (detached), rejected_logps (detached),
            chosen/rejected mean token weights and effective token counts
        """
```

Replace with:

```python
        """Forward pass for Ra-DPO: computes log-prob margins, KL, and CVaR risk ratios.

        Returns a 15-tuple:
            chosen_logps_margin, rejected_logps_margin,
            chosen_weighted_position_kl, rejected_weighted_position_kl,
            chosen_raw_position_kl, rejected_raw_position_kl,
            chosen_position_risk_ratio, rejected_position_risk_ratio,
            chosen_logps (detached), rejected_logps (detached),
            chosen_mean_token_weight, rejected_mean_token_weight,
            chosen_effective_token_count, rejected_effective_token_count,
            georisk_stats (dict[str, scalar]; empty unless mode='georisk')
        """
```

- [ ] **Step 9.5: Verify file parses**

```bash
python -c "import ast; ast.parse(open('scripts/run_distill_dpo.py').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 9.6: Commit**

```bash
git add scripts/run_distill_dpo.py
git commit -m "feat(georisk): build branch_sign + GeoRiskConfig in radpo_concatenated_forward

branch_sign is [+1 for chosen rows, -1 for rejected]; GeoRiskConfig is
built once per step from self.args when mode='georisk'. Returns a
15-tuple with georisk_stats appended.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Thread `georisk_stats` into `get_batch_loss_metrics` logging

**Files:**
- Modify: `scripts/run_distill_dpo.py:1819-1852` (Ra-DPO branch inside `get_batch_loss_metrics`)

- [ ] **Step 10.1: Update the radpo_concatenated_forward unpack and add georisk_stats logging**

In `scripts/run_distill_dpo.py:1819-1826`, the current code is:

```python
        if self.args.radpo_weight > 0.0001:
            (chosen_logps_margin, rejected_logps_margin,
             chosen_weighted_position_kl, rejected_weighted_position_kl,
             chosen_raw_position_kl, rejected_raw_position_kl,
             chosen_position_risk_ratio, rejected_position_risk_ratio,
             radpo_chosen_logps, radpo_rejected_logps,
             chosen_mean_token_weight, rejected_mean_token_weight,
             chosen_effective_token_count, rejected_effective_token_count) = self.radpo_concatenated_forward(model, batch)
```

Replace with:

```python
        if self.args.radpo_weight > 0.0001:
            (chosen_logps_margin, rejected_logps_margin,
             chosen_weighted_position_kl, rejected_weighted_position_kl,
             chosen_raw_position_kl, rejected_raw_position_kl,
             chosen_position_risk_ratio, rejected_position_risk_ratio,
             radpo_chosen_logps, radpo_rejected_logps,
             chosen_mean_token_weight, rejected_mean_token_weight,
             chosen_effective_token_count, rejected_effective_token_count,
             georisk_stats) = self.radpo_concatenated_forward(model, batch)
```

- [ ] **Step 10.2: Append the georisk_stats threading after the existing radpo_mu metric**

Find the line:

```python
            metrics[f"{prefix}radpo_mu"] = float(getattr(self, "_current_radpo_mu", self.args.radpo_confidence_level))
```

Immediately after it, append:

```python
            # GeoRiskKD diagnostics (empty dict when mode != "georisk").
            for _k, _v in georisk_stats.items():
                metrics[f"{prefix}{_k}"] = _v.detach().cpu() if hasattr(_v, "detach") else _v
```

- [ ] **Step 10.3: Verify file parses**

```bash
python -c "import ast; ast.parse(open('scripts/run_distill_dpo.py').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 10.4: Commit**

```bash
git add scripts/run_distill_dpo.py
git commit -m "feat(georisk): log georisk_stats into get_batch_loss_metrics

Unpacks the new 15th element of radpo_concatenated_forward and writes
each georisk/* scalar into the metrics dict with the train/eval prefix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Integration smoke — full test suite + py_compile

**Files:**
- No new files.

- [ ] **Step 11.1: Run the full unit-test suite**

```bash
pytest tests/test_georisk_features.py -v
```

Expected: 26 passed (25 from earlier + 1 from_args integration test in Task 7).

- [ ] **Step 11.2: Verify `scripts/run_distill_dpo.py` byte-compiles**

```bash
python -m py_compile scripts/run_distill_dpo.py && echo "py_compile OK"
```

Expected: `py_compile OK` (no SyntaxError, no NameError).

- [ ] **Step 11.3: Verify the import surface end-to-end**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from utils.georisk_features import (
    GeoRiskConfig, masked_mean, masked_zscore,
    project_reference_topk_with_labels, compute_branch_local_alignment,
    compute_relevance_salience, compute_instability_proxy,
    softmax_with_budget, compute_georisk_token_weights,
)
print('all imports OK')
"
```

Expected: `all imports OK`.

- [ ] **Step 11.4: No commit required** (this is a verification gate; nothing changed on disk).

If any step fails, fix it in the file that broke and re-run from Step 11.1.

---

## Task 12: Recipe `recipes/qwen3-1.7b-ultrafeedback/riskKD.yaml` (baseline)

**Files:**
- Create: `recipes/qwen3-1.7b-ultrafeedback/riskKD.yaml`

- [ ] **Step 12.1: Write the recipe**

Create `recipes/qwen3-1.7b-ultrafeedback/riskKD.yaml`:

```yaml
# riskKD baseline for the Qwen3 1.7B student / Qwen3 8B DPO teacher pair.
# Pure Ra-DPO (radpo_weight=1, dpo_weight=0), no token-weighting (mode='none').
# This is the unweighted baseline against which GeoRiskKD is measured.

# Model arguments
model_name_or_path: vukien2301/qwen3-1.7b-deita-sft-student
ref_model_name_or_path: vukien2301/qwen3-8b-ultrafeedback-dpo-teacher
chat_template: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
torch_dtype: bfloat16

# Data
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
- train_prefs
preprocessing_num_workers: 12

# Trainer
bf16: true
do_eval: false
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
learning_rate: 5.0e-7
log_level: info
logging_steps: 5
lr_scheduler_type: constant
max_length: 1024
max_prompt_length: 512
num_train_epochs: 1
optim: adamw_torch
output_dir: /workspace/riskKD/output/qwen3-1.7b-riskKD-baseline
per_device_train_batch_size: 4
push_to_hub: false
save_strategy: "epoch"
save_only_model: true
seed: 42
warmup_ratio: 0.1
dataset_num_proc: 16
beta: 0.1

# Loss weights (pure Ra-DPO)
sft_on_chosen: 0.0
sft_on_rejected: 0.0
dpo_weight: 0.0
distillation_weight: 0.0
chosen_distil_weight: 0.0
rejected_distil_weight: 0.0
adpa_weight: 0.0
qadapter_distil_weight: 0.0
kl_student_weight: 0.0
kl_penalty_weight: 0.0
precompute_ref_log_probs: false
reference_free: false
force_use_ref_model: true

# Ra-DPO
radpo_weight: 1.0
radpo_alpha: 0.5
radpo_confidence_level: 0.5
if_radpo2: false
is_split_risk_ratio: false
is_cal_risk_distribution_logps: false
radpo_keep_ref_model: true
radpo_mu_anneal_start: 0.99
radpo_mu_anneal_end: 0.5
radpo_mu_anneal_frac: 1.0

# Token weighting: OFF (baseline)
radpo_token_weight_mode: none
radpo_token_weight_target: all

# Logging
report_to: wandb
run_name: qwen3-1.7b-riskKD-baseline
```

- [ ] **Step 12.2: Verify the YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('recipes/qwen3-1.7b-ultrafeedback/riskKD.yaml'))" && echo "YAML OK"
```

Expected: `YAML OK`.

- [ ] **Step 12.3: Commit**

```bash
git add recipes/qwen3-1.7b-ultrafeedback/riskKD.yaml
git commit -m "feat(recipe): Qwen3 1.7B riskKD baseline (no token weighting)

Pure Ra-DPO with mode='none'. Student: vukien2301/qwen3-1.7b-deita-sft-student,
ref/teacher: vukien2301/qwen3-8b-ultrafeedback-dpo-teacher. This is the
baseline GeoRiskKD is measured against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Recipes `georiskKD.yaml` (target=risk) and `georiskKD_targetall.yaml` (target=all)

**Files:**
- Create: `recipes/qwen3-1.7b-ultrafeedback/georiskKD.yaml`
- Create: `recipes/qwen3-1.7b-ultrafeedback/georiskKD_targetall.yaml`

- [ ] **Step 13.1: Write `georiskKD.yaml` (target=risk default)**

```yaml
# GeoRiskKD — landscape-aware token weighting on top of Ra-DPO.
# Target = risk: weight is applied only to the Ra-DPO risk correction (delta).
# See docs/superpowers/specs/2026-05-13-georiskkd-design.md.

model_name_or_path: vukien2301/qwen3-1.7b-deita-sft-student
ref_model_name_or_path: vukien2301/qwen3-8b-ultrafeedback-dpo-teacher
chat_template: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
torch_dtype: bfloat16

dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
- train_prefs
preprocessing_num_workers: 12

bf16: true
do_eval: false
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
learning_rate: 5.0e-7
log_level: info
logging_steps: 5
lr_scheduler_type: constant
max_length: 1024
max_prompt_length: 512
num_train_epochs: 1
optim: adamw_torch
output_dir: /workspace/riskKD/output/qwen3-1.7b-georiskKD-risk
per_device_train_batch_size: 4
push_to_hub: false
save_strategy: "epoch"
save_only_model: true
seed: 42
warmup_ratio: 0.1
dataset_num_proc: 16
beta: 0.1

sft_on_chosen: 0.0
sft_on_rejected: 0.0
dpo_weight: 0.0
distillation_weight: 0.0
chosen_distil_weight: 0.0
rejected_distil_weight: 0.0
adpa_weight: 0.0
qadapter_distil_weight: 0.0
kl_student_weight: 0.0
kl_penalty_weight: 0.0
precompute_ref_log_probs: false
reference_free: false
force_use_ref_model: true

radpo_weight: 1.0
radpo_alpha: 0.5
radpo_confidence_level: 0.5
if_radpo2: false
is_split_risk_ratio: false
is_cal_risk_distribution_logps: false
radpo_keep_ref_model: true
radpo_mu_anneal_start: 0.99
radpo_mu_anneal_end: 0.5
radpo_mu_anneal_frac: 1.0

# GeoRiskKD
radpo_token_weight_mode: georisk
radpo_token_weight_target: risk
radpo_token_weight_normalize: false   # no-op for georisk; fixed-budget softmax already preserves Σw

radpo_georisk_top_k: 64
radpo_georisk_teacher_temperature: 1.0
radpo_georisk_lambda_risk: 1.0
radpo_georisk_lambda_relevance: 0.5
radpo_georisk_lambda_alignment: 0.5
radpo_georisk_lambda_kl: 1.0
radpo_georisk_lambda_instability: 0.0
radpo_georisk_lambda_unlearnability: 0.0
radpo_georisk_weight_tau: 1.0
radpo_georisk_weight_clip_min: 0.05
radpo_georisk_weight_clip_max: 3.0
radpo_georisk_stopgrad: true

report_to: wandb
run_name: qwen3-1.7b-georiskKD-risk
```

- [ ] **Step 13.2: Write `georiskKD_targetall.yaml` (target=all mirror)**

This is identical to Step 13.1 except for two fields. Create the file in full (do not write "same as above"):

```yaml
# GeoRiskKD — target=all mirror. Weights apply to BOTH the utility log-ratios
# and the Ra-DPO risk correction. Tests the large-gap variant from §9 of the
# v2 proposal.

model_name_or_path: vukien2301/qwen3-1.7b-deita-sft-student
ref_model_name_or_path: vukien2301/qwen3-8b-ultrafeedback-dpo-teacher
chat_template: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
torch_dtype: bfloat16

dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
- train_prefs
preprocessing_num_workers: 12

bf16: true
do_eval: false
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
learning_rate: 5.0e-7
log_level: info
logging_steps: 5
lr_scheduler_type: constant
max_length: 1024
max_prompt_length: 512
num_train_epochs: 1
optim: adamw_torch
output_dir: /workspace/riskKD/output/qwen3-1.7b-georiskKD-all
per_device_train_batch_size: 4
push_to_hub: false
save_strategy: "epoch"
save_only_model: true
seed: 42
warmup_ratio: 0.1
dataset_num_proc: 16
beta: 0.1

sft_on_chosen: 0.0
sft_on_rejected: 0.0
dpo_weight: 0.0
distillation_weight: 0.0
chosen_distil_weight: 0.0
rejected_distil_weight: 0.0
adpa_weight: 0.0
qadapter_distil_weight: 0.0
kl_student_weight: 0.0
kl_penalty_weight: 0.0
precompute_ref_log_probs: false
reference_free: false
force_use_ref_model: true

radpo_weight: 1.0
radpo_alpha: 0.5
radpo_confidence_level: 0.5
if_radpo2: false
is_split_risk_ratio: false
is_cal_risk_distribution_logps: false
radpo_keep_ref_model: true
radpo_mu_anneal_start: 0.99
radpo_mu_anneal_end: 0.5
radpo_mu_anneal_frac: 1.0

# GeoRiskKD — target=all
radpo_token_weight_mode: georisk
radpo_token_weight_target: all
radpo_token_weight_normalize: false

radpo_georisk_top_k: 64
radpo_georisk_teacher_temperature: 1.0
radpo_georisk_lambda_risk: 1.0
radpo_georisk_lambda_relevance: 0.5
radpo_georisk_lambda_alignment: 0.5
radpo_georisk_lambda_kl: 1.0
radpo_georisk_lambda_instability: 0.0
radpo_georisk_lambda_unlearnability: 0.0
radpo_georisk_weight_tau: 1.0
radpo_georisk_weight_clip_min: 0.05
radpo_georisk_weight_clip_max: 3.0
radpo_georisk_stopgrad: true

report_to: wandb
run_name: qwen3-1.7b-georiskKD-all
```

- [ ] **Step 13.3: Verify both YAMLs parse**

```bash
python -c "
import yaml
for p in ('recipes/qwen3-1.7b-ultrafeedback/georiskKD.yaml',
          'recipes/qwen3-1.7b-ultrafeedback/georiskKD_targetall.yaml'):
    yaml.safe_load(open(p))
    print(p, 'OK')
"
```

Expected: both files report OK.

- [ ] **Step 13.4: Commit**

```bash
git add recipes/qwen3-1.7b-ultrafeedback/georiskKD.yaml recipes/qwen3-1.7b-ultrafeedback/georiskKD_targetall.yaml
git commit -m "feat(recipe): Qwen3 1.7B GeoRiskKD recipes (target=risk + target=all)

georiskKD.yaml is the default (clean attribution, only the Ra-DPO risk
correction is weighted). georiskKD_targetall.yaml is the large-gap
variant from §9 of the v2 proposal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Update `CLAUDE.md` losses table

**Files:**
- Modify: `CLAUDE.md` (the losses table around the "How the trainer composes losses" section)

- [ ] **Step 14.1: Locate the losses table**

Open `CLAUDE.md` and find the row in the losses table for **Ra-DPO / CVaR (riskKD)**. It begins with `| **Ra-DPO / CVaR (riskKD)** | ...`.

- [ ] **Step 14.2: Append a new row below it for GeoRiskKD**

After the existing Ra-DPO/CVaR row, insert:

```markdown
| **GeoRiskKD token weighting** | `radpo_token_weight_mode="georisk"` + `radpo_georisk_lambda_*`, `radpo_georisk_top_k`, `radpo_georisk_weight_{tau,clip_min,clip_max}`, `radpo_georisk_stopgrad` | `compute_georisk_token_weights` in `utils/georisk_features.py` (called from `_radpo_get_batch_logps`); needs no DCKD precompute |
```

- [ ] **Step 14.3: Append a "Practical implication" sentence**

Find the existing "**Practical implication:**" paragraph in the same section. At the end of that paragraph, append one sentence:

```
GeoRiskKD layers on top of Ra-DPO by replacing the `kl_inv` token-weight branch with a fixed-budget softmax allocation over six branch-local features; see [`docs/superpowers/specs/2026-05-13-georiskkd-design.md`](docs/superpowers/specs/2026-05-13-georiskkd-design.md).
```

- [ ] **Step 14.4: Verify the file still renders sensibly**

```bash
wc -l CLAUDE.md
grep -n "GeoRiskKD" CLAUDE.md
```

Expected: line count grew by 1-2 lines; one or two GeoRiskKD hits.

- [ ] **Step 14.5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): add GeoRiskKD token weighting to losses table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (post-write checklist)

Before declaring the plan ready:

- [ ] **Spec coverage** — every requirement in the design spec has a task:
  - §3 locked decisions → Tasks 1–6 (helpers), 7 (config), 8–9 (wiring), 12–13 (recipes)
  - §4 architecture → Tasks 1–6 produce `utils/georisk_features.py`; Tasks 8–10 produce the trainer wiring
  - §5 config fields → Task 7
  - §6 data flow → Tasks 8 (call site inside `_radpo_get_batch_logps`) and 9 (`branch_sign` build)
  - §6.3 return contract → Tasks 8 (helper) and 9 (caller)
  - §7 features → Tasks 1 (mean/zscore), 2 (projection), 3 (alignment), 4 (relevance + instability), 6 (full assembly)
  - §8 score + allocation → Task 6 (assembly), Task 5 (softmax budget)
  - §9 edge cases → Task 6 (all-masked, nonfinite, lambda=0 short-circuit), Task 8 (mutual exclusion)
  - §10 performance budget → guarded by `teacher_temperature=1.0` default in recipes (Task 13)
  - §11 diagnostics → Task 6 (stats dict), Task 10 (threading)
  - §12 testing → Tasks 1–6 cover every helper invariant + Task 11 integration smoke
  - §13 rollout → Tasks 12–14
  - §14 experiment matrix → Tasks 12–13 (riskKD baseline + 2 georisk variants); ablations are CLI overrides
- [ ] **No placeholders** — no "TBD" / "TODO" / "implement later" / "similar to Task N" appears anywhere.
- [ ] **Type consistency** — function names match across tasks: `masked_mean`, `masked_zscore`, `project_reference_topk_with_labels`, `compute_branch_local_alignment`, `compute_relevance_salience`, `compute_instability_proxy`, `softmax_with_budget`, `compute_georisk_token_weights`, `GeoRiskConfig`, `GeoRiskConfig.from_args`. All field names in `radpo_georisk_*` match between Task 7 (config) and Task 13 (recipes).

---

## Operational notes for the executor

- Run all `pytest` commands from the repo root (`/Users/kienvu/Desktop/research-papers/riskKD`).
- The repo's main venv is at `/workspace/riskKD/.venv` **on the remote box**; for local development, ensure your environment has `torch`, `pytest`, and `pyyaml` (the existing repo deps cover these). If running tests locally on macOS without CUDA, all tests in this plan are CPU-only by design.
- After Task 11 succeeds, the implementation is logically complete on disk; Tasks 12–14 only add config files and docs. Recipes can be authored before the wiring lands in principle, but the suggested order keeps the integration smoke as the last code-touching checkpoint.
- Each commit message uses the `feat(...)` / `docs(...)` prefixes consistent with the existing repo style.

---

## Out of scope (deferred to v2)

- Live (non-precomputed) full-vocab teacher pass for the alignment feature.
- Real teacher value-change `ψ` (TVKD-style shifted soft value).
- Dropout-variance or perturbation-based sharpness for `I_t`.
- LoRA-space parameter-gradient alignment diagnostic.
- λ curricula across training.
- Reuse of `compute_georisk_token_weights` for TVKD `qadapter_*` or DCKD distillation losses.
