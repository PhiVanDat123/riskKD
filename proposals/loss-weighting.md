# Proposal: Adaptive weighting in the riskKD / Ra-DPO loss

## Motivation

In the 1B-student × 8B-teacher run on `pvdhihihi/ultra-feedback`, the Ra-DPO objective was dominated by cross-architecture KL noise rather than preference signal:

| Step | `loss/radpo` | `radpo_kl/chosen` | `radpo_rewards/margins` | `radpo_rewards/accuracies` |
|---|---|---|---|---|
| 30  | 10.06 | 174 | -6.5  | 0.375 |
| 196 | 4.64  | 1561 | -0.31 | 0.50 |
| 526 | 4.64  | 1215 | -0.31 | 0.375 |
| 855 | 0.003–6.6 | 1206 | +14.4 | 0.625–1.0 |
| final (902) | ~0.4 | 833 | +3.77 | 0.75 |

Loss converged late and noisily because the 1B policy cannot, in distribution, match an 8B reference's logit landscape. The current trainer applies one **scalar weight per loss component** (`radpo_weight`, `dpo_weight`, `qadapter_distil_weight`, ...) — every sample contributes equally inside that term. We want to add **finer-grained weights** so the loss focuses on samples / tokens / regimes that carry usable signal.

## Strategies (ordered: cheapest first → most invasive)

### 1. Teacher-temperature softening *(single hyperparameter, ~10-line change)*

Apply temperature `T > 1` to the teacher distribution before computing KL / CVaR:
```
p_ref^T(t) = softmax(logits_ref(t) / T)
```
- High `T` smooths the teacher's peaks → smaller KL, easier student fit.
- Standard distillation trick (Hinton et al. 2015).
- **Best single-knob fix to the cross-size problem we hit.**
- Code site: `_radpo_get_batch_logps` (`scripts/run_distill_dpo.py:519`), divide `reference_logits` by `T` before the softmax.
- New YAML field: `radpo_teacher_temperature: float = 1.0`.

### 2. Per-sample KL-distance weighting

Downweight samples whose policy/reference distributions are far apart — these are the noise-dominated samples in cross-size training.
```
w_i = exp(-α · KL_i)        # multiplicative dampening
or:    1 / (1 + KL_i/KL_0)   # bounded
```
- Applied to each sample's contribution before the final reduce-mean inside `radpo_loss_fn`.
- Hyperparams: `α` (or `KL_0`).
- Pure stabilizer — does not change loss direction, only magnitude.
- Code site: `radpo_loss_fn` in `scripts/run_distill_dpo.py:579`.

### 3. Teacher-margin weighting

Emphasize pairs the teacher itself confidently prefers; uncertain pairs (where teacher gives chosen ≈ rejected) get downweighted.
```
m_i = logp_ref(chosen_i) - logp_ref(rejected_i)
w_i = sigmoid(β_w · m_i)
```
- Cleaner preference signal: hard pairs from a confused teacher are noise.
- Hyperparam: `β_w` (sharpness).
- Cheap because `logp_ref(chosen)` and `logp_ref(rejected)` are already computed.
- Code site: `radpo_loss_fn`, after computing `chosen_logps_margin` / `rejected_logps_margin`.

### 4. Token-level weighting inside Ra-DPO

The current `_calculate_cvar_radpo` uses a hard binary mask `distribution > VaR`. Soft alternatives:

**(a) Position decay** — later tokens (closer to the answer in chat templates) get more weight:
```
w_t = γ^(L-t)     # γ ∈ (0,1]
```

**(b) Magnitude-based soft mask** — replace the binary CVaR mask with a soft variant:
```
soft_mask_t = sigmoid(λ · (distribution_t - VaR))
```

**(c) Token-margin mask** — only count tokens where teacher prefers chosen over rejected at that position:
```
mask_t = (logp_ref(c_t) > logp_ref(r_t))
```

- Code site: inside `_calculate_cvar_radpo` (`scripts/run_distill_dpo.py:452`).
- Hyperparams: `γ` (decay), or `λ` (soft mask sharpness).
- Most invasive of the four, but the most flexible — directly modifies how risk is computed per position.

### 5. Adaptive scheduling between loss components

Today every weight in the YAML is constant for the whole run. Two scheduled variants:

**(a) Annealing** — start grounded in SFT, transition toward DPO:
```
sft_weight(t)  = (1 - t/T) · sft_init
radpo_weight(t) = (t/T) · radpo_final
```
Implementable as a callback in `DistillTrainer`.

**(b) Gradient-magnitude balancing (GradNorm/DWA)** — at each logging step, rescale weights so each active loss component contributes a comparable gradient magnitude. Prevents one term (e.g. Ra-DPO) from dominating SFT or KL.

- Code site: `DistillTrainer.get_batch_loss_metrics` (`scripts/run_distill_dpo.py:1636`), wrap each `losses = losses + w * term` with a dynamic `w(step)`.

### 6. Focal-loss-style margin reweighting

For DPO/Ra-DPO sigmoid losses, focus gradient on hard examples (where current policy is wrong):
```
w_i = (1 - σ(β · m_i))^γ
```
- `γ = 0` → standard DPO;  `γ > 0` → focal — large gradient only when policy is uncertain or wrong.
- Hyperparam: `γ`.
- Code site: inside `radpo_loss_fn`, before the final `-log σ(...)` step.

## Implementation priority

| # | Strategy | Effort | Expected impact in our cross-size case |
|---|---|---|---|
| **1** | Teacher temperature | **small** (single param + division) | **high** — directly attacks KL explosion |
| **2** | KL-distance weighting | small | high — stabilizes |
| 3 | Teacher-margin weighting | small | medium — improves signal/noise |
| 6 | Focal-loss reweighting | small | medium — only helps once policy is reasonable |
| 4 | Token-level CVaR softening | medium | medium-high — extends the actual Ra-DPO formulation |
| 5 | Loss-component scheduling | medium | situational — useful when terms compete |

**Recommended pilot:** combine #1 (teacher temperature `T=2.0`) + #2 (KL-distance weighting `α=1e-3`) on the same 1B/8B Ra-DPO setup. Both are cheap to implement and target the exact failure mode we observed. If the pilot stabilizes training, add #3 and #4.

## Open questions

- For #1, what teacher temperature is "too soft"? At `T → ∞`, the teacher is uniform — no signal at all. Empirical sweep over `{1.0, 1.5, 2.0, 3.0, 5.0}`.
- For #2, the right `α` depends on the typical KL scale. We saw KL ~ 800–1500 in the cross-size run; with `α = 1e-3`, the weight is `e^(-1) ≈ 0.37` at KL=1000. Need to log per-sample KL distribution to pick `α` properly.
- All these are *training-time* knobs; none change the eval pipeline.
- Should we expose all of these as YAML fields, or hard-code defaults and only surface the ones that matter? Decision point after pilot.

## YAML fields to add (proposed)

```yaml
# Teacher softening
radpo_teacher_temperature: 1.0   # >1 softens the teacher distribution

# Per-sample weighting
radpo_kl_weight_alpha: 0.0       # 0 disables; >0 enables exp(-α·KL) downweighting
radpo_margin_weight_beta: 0.0    # 0 disables; >0 enables sigmoid(β·teacher_margin) emphasis

# Focal-style preference loss
radpo_focal_gamma: 0.0           # 0 = standard sigmoid loss; >0 focuses gradient on hard examples

# Token-level CVaR softening
radpo_position_decay_gamma: 1.0  # 1.0 = uniform across positions; <1 emphasizes later tokens
radpo_soft_mask_lambda: 0.0      # 0 = original binary mask; >0 = soft sigmoid mask

# Adaptive schedules (advanced)
radpo_anneal_start_frac: 0.0     # 0 = no anneal; >0 ramps radpo_weight from 0 over this fraction of steps
```

All default to "off" (numerically equivalent to current behavior), so the YAML is backwards-compatible.
