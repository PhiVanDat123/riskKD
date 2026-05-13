# GeoRiskKD — Implementation Design

**Status:** approved design, pre-implementation
**Date:** 2026-05-13
**Paper proposal:** [`georiskkd_landscape_aware_risk_weighting.md`](../../../georiskkd_landscape_aware_risk_weighting.md)
**Codebase entry point:** `scripts/run_distill_dpo.py`
**Target recipe family:** `recipes/qwen3-1.7b-ultrafeedback/`

---

## 1. Goal

Implement the GeoRiskKD token-weighting scheme from the proposal as a new mode of the
existing Ra-DPO / RiskKD trainer. The trainer already accepts a per-token weight tensor
via `radpo_token_weight_mode={"none","kl_inv"}` at
`scripts/run_distill_dpo.py:558-617`. GeoRiskKD adds a third mode, `"georisk"`, that
computes `w_t` from six forward-only features and a fixed-budget softmax allocation,
matching §6 of the proposal in full.

Out of scope for v1: gradient-alignment `A_t` (paper §6.3 ideal form), dropout-variance
`S_t` (paper §6.4 second proxy), EMA-deviation `S_t`, curriculum schedules on the λs.

---

## 2. Decisions locked in brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| v1 feature set | All 6 features from §6 | User-selected. |
| Teacher distribution source | Precomputed top-K from the DCKD pipeline | Reuses existing `teacher_chosen_probs`/`teacher_rejected_probs` columns. No live teacher in memory. |
| `m_t` form | `m_t = \|ψ_t\|` per-branch (drop cross-branch formula) | Avoids the `T_w ≠ T_l` alignment problem. |
| `target` default | `risk` (and ship `target=all` as a mirror YAML) | Matches paper §9 recommendation. |
| `S_t` proxy | Batch-relative `\|ℓ_t − batch_mean(ℓ_t)\|` | Stateless; ~0 overhead; matches paper §6.4 "even cheaper proxy". |
| `A_t` formula | Top-K cosine in probability space (no full-vocab reconstruction) | Forward-only; respects precomputed-top-K limitation. |
| Feature normalization | Per-sequence z-score over valid tokens | Keeps softmax-budget allocation scale-stable across features. |
| Architecture | Approach (B): helper in `radpo_concatenated_forward`, precomputed weights passed into `radpo_loss_fn` | Lean loss function, unit-testable. |

---

## 3. Architecture

```
recipes/qwen3-1.7b-ultrafeedback/
  georiskKD.yaml              # target=risk default
  georiskKD_targetall.yaml    # target=all mirror

scripts/run_distill_dpo.py
  CustomDPOConfig:            # +13 fields (one new mode, twelve hyperparams)
  radpo_concatenated_forward: # +~80 LOC: calls compute_georisk_token_weights, threads w_t to loss
  radpo_loss_fn:              # +~30 LOC: "georisk" arm consuming a precomputed token_weights tensor

utils/georisk_features.py     # NEW (~250 LOC) — pure forward-only functions:
  compute_psi_t               # teacher value change (m_t signal)
  compute_alignment_t         # top-K cosine
  compute_sharpness_t         # batch-relative |ℓ_t − mean|
  compute_learnability_t      # student NLL
  zscore_per_seq              # per-sequence z-score with mask
  softmax_with_budget         # softmax · |M_y|, then clamp, then renormalize, then stop-grad
  compute_georisk_token_weights  # top-level orchestrator called from the trainer

tests/test_georisk_features.py  # NEW — frozen-tensor unit tests
```

`r_t` and `D_t` are not re-implemented; they come from `per_position_risk_ratio` and
`per_position_kl`, both already produced by `radpo_concatenated_forward`.

---

## 4. CustomDPOConfig fields

Extends the existing `radpo_token_weight_*` block:

```python
# existing (extended)
radpo_token_weight_mode: Literal["none", "kl_inv", "georisk"] = "none"
radpo_token_weight_target: Literal["all", "risk"] = "all"
radpo_token_weight_normalize: bool = False   # no-op for georisk mode (softmax-budget already enforces Σw = |M_y|); honored only by kl_inv

# NEW
radpo_georisk_teacher_temperature: float = 2.0   # π_T^τ used for ψ + A_t (paper §6.1)
radpo_georisk_softvalue_type: Literal["entropy", "lse", "sum"] = "entropy"
radpo_georisk_lambda_risk:         float = 1.0   # λ_r · ẑ(r_t)
radpo_georisk_lambda_margin:       float = 1.0   # λ_m · ẑ(m_t)
radpo_georisk_lambda_alignment:    float = 0.5   # λ_A · ẑ(A_t)
radpo_georisk_lambda_kl:           float = 1.0   # − λ_D · ẑ(D_t)
radpo_georisk_lambda_sharpness:    float = 0.25  # − λ_S · ẑ(S_t)
radpo_georisk_lambda_learnability: float = 0.0   # − λ_N · ẑ(N_t) — off by default per §7
radpo_georisk_weight_tau:          float = 1.0
radpo_georisk_weight_clip_min:     float = 0.05
radpo_georisk_weight_clip_max:     float = 3.0
radpo_georisk_stopgrad:            bool  = True
```

Setting any `lambda_*` to `0.0` short-circuits the corresponding feature computation
(important for ablations and for `A_t`, which is the most expensive feature).

---

## 5. Data flow

For each branch `y ∈ {y_w, y_l}` (chosen and rejected computed independently, then both
weight tensors handed to `radpo_loss_fn`):

```
INPUTS (already in radpo_concatenated_forward):
  student_logits             [B, T, V]
  teacher_*_probs            [B, T, K, 2]    # (top-K prob, top-K idx) — from DCKD precompute
  per_token_logps            [B, T]          # log π_θ(y_t|s_t)
  per_position_kl            [B, T]          # = D_t
  per_position_risk_ratio    [B, T]          # = r_t
  loss_mask                  [B, T]

STEP 1 — student probs at teacher's top-K vocab ids
  idx            = teacher_probs[..., 1].long()                    # [B, T, K]
  student_topkp  = softmax(student_logits/temp).gather(-1, idx)    # [B, T, K]
  teacher_topkp  = teacher_probs[..., 0]                            # [B, T, K]

STEP 2 — features (torch.no_grad()), each shape [B, T]
  r_t  = per_position_risk_ratio
  D_t  = per_position_kl
  N_t  = -per_token_logps
  S_t  = |N_t - masked_mean(N_t, dim=1)|
  A_t  = cosine(teacher_topkp, student_topkp, dim=-1)        # only if lambda_alignment > 0; else 0

STEP 3 — ψ_t (teacher value change)
  V_t   = softvalue(teacher_topkp, student_topkp, type=softvalue_type)
        # entropy: V_t = -Σ_k p_T_k · log p_T_k                     (student_topkp ignored)
        # lse:     V_t = log Σ_k exp(log p_T_k + log K)             (student_topkp ignored)
        # sum:     V_t = Σ_k p_T_k · log(p_T_k / p_θ_k)             ← ablation only; near-degenerate with D_t
  V_tp1 = shift_left(V_t, fill=0.0)                                # terminal sentinel
  ψ_t   = Φ_μ(V_tp1) - Φ_μ(V_t)        # Φ_μ reuses Ra-DPO's existing nested risk operator
  m_t   = |ψ_t|

STEP 4 — score and allocation
  z = zscore_per_seq(stack([r,m,A,D,S,N]), loss_mask)              # [6, B, T]
  q = λ_r·z[0] + λ_m·z[1] + λ_A·z[2] - λ_D·z[3] - λ_S·z[4] - λ_N·z[5]
  q = q.masked_fill(~loss_mask, -inf)
  w = softmax(q/τ, dim=-1) * loss_mask.sum(-1, keepdim=True)
  w = clamp(w, w_min, w_max)
  w = w * loss_mask.sum(-1, keepdim=True) / (w*loss_mask).sum(-1, keepdim=True).clamp_min(1e-8)
  if stopgrad: w = w.detach()

OUTPUT
  w_t per branch -> radpo_loss_fn(token_weights=w_t, token_weight_target=...)
```

`Φ_μ` is the same nested risk operator already used inside `_calculate_cvar_radpo` /
`_cal_risk_distribution_logps_radpo`. We do not reimplement it — we expose it as a
helper or inline-call the existing function with the right shape.

---

## 6. Error handling and edge cases

| Case | Behavior |
|---|---|
| All-masked row (`loss_mask.sum(-1) == 0`) | Skip GeoRiskKD; fall back to uniform `w_t = 1`. Avoids divide-by-zero in z-score and `softmax(−∞,...)`. |
| Zero-variance feature in a sequence | Clamp `σ_seq ≥ 1e-8`; the feature contributes 0 to `q_t`. |
| Cosine on near-zero vectors | Add `1e-8` to denominator; clamp `A_t` to `[−1, 1]`. |
| NaN/Inf in any feature | Replace with 0; increment `georisk/nonfinite_count` counter. |
| `teacher_*_probs` missing from batch | Hard `ValueError` at trainer init: `"radpo_token_weight_mode='georisk' requires precomputed teacher_*_probs columns; build them with run/dckd.sh"`. |
| `top-K` mismatch across rows | Assert `K == dataset.K`; the precompute pipeline already enforces a fixed K. |
| `λ_X = 0` | Short-circuit feature computation (no `gather`, no `cosine`, no `softvalue`). |
| `stopgrad=True` (default) | `w_t = w_t.detach()`; test asserts no gradient leaks into `teacher_*_probs`. |
| DeepSpeed Zero-3 | All ops are intra-rank (per-sequence reductions). No new collective communication. |

---

## 7. Performance budget

For Qwen3-1.7B student × Qwen3-8B teacher, batch B=8, seq T=2048, K=8:

| Op | Tensor shape | Cost |
|---|---|---|
| `gather` for student top-K probs | `[B, T, V]` → `[B, T, K]` | ~1 GB scratch; fused gather |
| 6 features | `[B, T]` each | negligible |
| z-score + softmax | `[6, B, T]` then `[B, T]` | negligible |
| `Φ_μ` on V_t and V_{t+1} | `[B, T]` each | already on the hot path for Ra-DPO |
| Total step-time overhead | | ~2-4% of the existing Ra-DPO step |
| Total memory overhead | | <5% (one `[B, T, K]` scratch) |

No new model weights, no extra forward pass.

---

## 8. Diagnostics logged per step

```
georisk/r_mean, georisk/r_top_weighted        # top_weighted = mean over top-10% weighted tokens
georisk/m_mean, georisk/m_top_weighted
georisk/A_mean, georisk/A_top_weighted
georisk/D_mean, georisk/D_top_weighted
georisk/S_mean, georisk/S_top_weighted
georisk/N_mean, georisk/N_top_weighted
georisk/token_weight_entropy
georisk/effective_tokens
georisk/top10_weight_mass
georisk/weight_chosen_mean
georisk/weight_rejected_mean
georisk/nonfinite_count
```

Integrates with the existing W&B logging (`report_to: wandb` already wired by
`run_box/box_qwen_wandb_setup.sh`).

---

## 9. Testing

`tests/test_georisk_features.py` (new), CPU-only, ~150 LOC:

1. **Frozen-tensor goldens** for each of the six feature functions: hand-built
   `[2, 5, 8]` inputs with hand-computed expected outputs to `1e-5` tolerance.
2. **Aggregation invariants** on the orchestrator output:
   - `(w * mask).sum(-1) == mask.sum(-1)` to `1e-5` (budget preserved post-clamp).
   - `w_t.requires_grad is False` when `stopgrad=True`.
   - `w_t ∈ [w_min, w_max]` on valid positions.
3. **Edge cases**: all-masked row → uniform fallback; zero-variance feature → no NaN;
   NaN/Inf in feature → 0 + counter incremented.
4. **`λ=0` short-circuit**: monkeypatch `cosine` to raise; verify it is *not* called
   when `lambda_alignment=0`.
5. **End-to-end smoke** (skipped if no GPU): one training step on a 2-example synthetic
   batch; assert loss is finite, weights sum to budget, and gradients flow only into
   model params.

Tests run via `pytest tests/test_georisk_features.py`. No CI changes — this is
research code.

---

## 10. Rollout plan

**v1 deliverables in this implementation:**

- `utils/georisk_features.py` (new)
- `scripts/run_distill_dpo.py` (edited): `CustomDPOConfig` fields,
  `radpo_concatenated_forward` calls, `radpo_loss_fn` `"georisk"` arm
- `recipes/qwen3-1.7b-ultrafeedback/georiskKD.yaml` (new, `target=risk`)
- `recipes/qwen3-1.7b-ultrafeedback/georiskKD_targetall.yaml` (new, `target=all` mirror)
- `tests/test_georisk_features.py` (new)
- Updated `CLAUDE.md` losses table (one row)

**Not in this implementation, separate operational steps before running on the box:**

- Build the Qwen3 `ultrafeedback-dckd` precomputed dataset:
  `bash run/dckd.sh` with the Qwen3-8B DPO teacher (`vukien2301/qwen3-8b-ultrafeedback-dpo-teacher`).
  Estimated ~30–60 min on 2× B200 with the existing `fast_merge_dckd.py` from
  `run_box/box_setup2.sh`.
- Recipe path-rewriting (the existing YAMLs still carry stale `/home/minchan.kwon/...`
  paths — handled by an extension of `box_setup2.sh`).

**Experiment matrix** (after v1 lands, paper §15 + §21):

| # | YAML | Notes |
|---|---|---|
| 0 | existing `riskKD.yaml` | baseline |
| 1 | `georiskKD.yaml`, target=risk, λ_N=0 | §7 starting point |
| 2 | `georiskKD_targetall.yaml`, target=all, λ_N=0 | large-gap variant |
| 3 | `georiskKD.yaml` − λ_r=0 | does risk allocation matter |
| 4 | `georiskKD.yaml` − λ_m=0 | does ψ matter |
| 5 | `georiskKD.yaml` − λ_A=0 | does geometry beat KL |
| 6 | `georiskKD.yaml` − λ_S=0 | flatness term |
| 7 | `georiskKD.yaml` − λ_D=0 | trust penalty |

Only #1 and #2 ship as committed YAMLs; the ablation variants are `cp + sed` at
experiment time.

---

## 11. Open questions deferred to v2

- Gradient-alignment `A_t` (paper §6.3 ideal form) — requires per-token grad probes;
  out of scope for forward-only v1.
- Curriculum on the λs (paper §18 fix 3) — schedule the λs over training; v2.
- Live (non-precomputed) teacher distribution path — useful if A_t needs full vocab;
  v2.
- Reuse of `compute_georisk_token_weights` for TVKD's `qadapter_*` or DCKD's
  `distillation_weight` paths — possible after v1 with a small refactor.

---

## 12. References

- Paper proposal: `georiskkd_landscape_aware_risk_weighting.md` (in repo root)
- Existing token-weight machinery: `scripts/run_distill_dpo.py:558-617` (`radpo_loss_fn`'s
  `kl_inv` branch)
- Ra-DPO core: `scripts/run_distill_dpo.py:486` (`_calculate_cvar_radpo`), `:514`
  (`_cal_risk_distribution_logps_radpo`)
- TVKD soft value: `qadapter_softvalue_type` switch documented in `CLAUDE.md` (project root)
