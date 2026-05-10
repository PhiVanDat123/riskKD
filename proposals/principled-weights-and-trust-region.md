# Proposal (P0): Per-token weights as importance correction + per-token trust region

Status: **methods reframing** (#1, #2) — no new code needed for the reframing itself; the `kl_inv` token weighting already implemented in `scripts/run_distill_dpo.py` is the empirical instantiation. The annealed-μ piece (#5) is implemented separately (`radpo_mu_anneal_*`).

## Background — what's currently ad-hoc

We added `w_t = exp(-α · KL_t)` per-token weighting to the Ra-DPO/ARR loss to stop the cross-size (1B student / 8B teacher) explosion (KL ~1000+ → ~30, train_loss 5.4 → 0.98, reward margins flipped from negative to positive). It works, but as written it's a heuristic with no grounding in the ARR derivation. P0 turns it into a derived component.

## #1 — The per-token weights `w_t` ARE the importance-sampling correction the framework requires

**Observation.** The paper's objective already carries per-token weights that are never specified:
- Eq. 10 (utility gap): `u = Σ_i w^w_i · β log(π_θ(y^w_i|·)/π_tch(y^w_i|·)) − Σ_j w^l_j · β log(π_θ(y^l_j|·)/π_tch(y^l_j|·))`
- Eq. 12 (sequential risk ratio): `D_SeqRR(π_tch ‖ π_θ) = Σ_t w_t · Φ^μ( E_{z∼π_tch}[ log(π_tch(z|s̃_t)/π_θ(z|s̃_t)) ] )`

In the paper these `w_t` are placeholders ("we may down-weight..."). **Claim: they should be the importance weights induced by the augmented Pb-MDP, and a KL-trust-region assumption makes them `exp(-α · KL_t)`.**

**Why an IS weight is needed at all.** The closed-form optimal policy (Eq. 6) is `π*_θ(z|s̃_t) ∝ π_tch(z|s̃_t) · exp(Q̃_{π_tch}(s̃_t,z)/β)` — the teacher `π_tch` is the *reference measure*. The risk operator `Φ^μ` and the value function `Ṽ_{π_tch}` are defined as expectations **under `π_tch`** (Eq. 2–3). But during training we only observe trajectories from the offline data and evaluate `π_θ`. To form an unbiased estimate of any `E_{z∼π_tch}[f(s̃_t,z)]` using the on-policy state `s̃_t = [x, y_{<t}]` reached under `π_θ` (or under the data policy), the per-state contribution must be reweighted. The natural reweighting is the *state-occupancy ratio* between the teacher-induced and the realized trajectory distributions up to step `t`. Concretely, for a trajectory prefix `y_{<t}`:

```
ρ_t  =  ∏_{k<t}  π_tch(y_k | x, y_{<k}) / π_θ(y_k | x, y_{<k})           (telescoping IS weight)
```

A bare product IS weight has notoriously high variance. The standard remedy (used throughout off-policy RL and offline-to-online distillation) is to bound it. If we assume a **per-step trust region** — `KL( π_θ(·|s̃_t) ‖ π_tch(·|s̃_t) ) ≤ ε` — then the per-step log-ratio `log ρ_t/ρ_{t-1}` is controlled by `KL_t`, and a first-order / exponential-tilting bound gives

```
w_t  ≈  exp( −α · KL_t )            with α absorbing the per-step trust-region constant
```

i.e. **down-weight a token by how far the student's distribution has drifted from the teacher's at that state.** This is exactly the `kl_inv` weighting we implemented — but now it's *the* weight the objective was missing, not an add-on.

**Consequences for the writeup:**
- Eq. 10 and Eq. 12 get a *definition* for `w_t` instead of a placeholder: `w_t = exp(−α KL_t)` (or the un-bounded `ρ_t`, with `α` as a stabilization knob; `α=0` recovers the naive IS estimator, `α→∞` recovers a behavior-cloning-like regime).
- The "large-capacity-gap" failure of vanilla ARR is *predicted* by the theory: when student and teacher are far apart, `KL_t` is large, the naive IS weights blow up, and the estimator of `Φ^μ(Ṽ_{π_tch})` becomes meaningless — which is precisely the `radpo_kl ~ 1000`, sign-flipping-margins behavior we observed at `α=0`.
- The empirical fix (KL 1000+ → ~30, loss converges, margins positive, +0.70 avg over the lm-eval suite, +2.3 GSM8K / +2.4 ARC) is the *evidence* that the IS-correction interpretation is the right one.

**One thing to be careful about in the derivation:** the trust-region bound `w_t = exp(−α KL_t)` is heuristic-grade unless you commit to a specific bounding lemma. Two clean options: (a) treat `α` as a Lagrange multiplier on the per-step trust region and present `w_t = exp(−α KL_t)` as the resulting tilted weight (clean, standard); (b) present it as a *bounded* IS estimator `w_t = min(ρ_t, c)` with the `exp(−α KL)` form as a smooth surrogate. Option (a) reads better in a methods section.

## #2 — Per-token trust region: unify the token weighting with ARR's KL regularizer

ARR already has a global KL term (Eq. 5): `max E[ r + ψ − β · KL(π_θ(·|s̃_t) ‖ π_tch(·|s̃_t)) ]`. The `exp(−α KL_t)` token weighting is *also* a function of the same per-step KL. Right now they're two separate mechanisms applied to the same quantity. **Unify them: there is one per-token trust-region knob.**

Interpretation: ARR wants the student to (i) follow the teacher's risk-adjusted value guidance and (ii) not stray too far from the teacher. Both should be *localized*: at a state where the student is already close to the teacher (`KL_t` small) — trust the guidance fully and apply little regularization. At a state where the student has drifted (`KL_t` large) — the guidance is unreliable (the IS weight is huge / the value estimate is off), so *both* down-weight that token's contribution to the objective *and* (optionally) penalize it more. A single per-step "trust" function `τ(KL_t)`:
- multiplies the utility-gap and risk-correction contributions: `w_t = τ(KL_t)` (this is #1's weight),
- and, if you want, scales the local KL penalty: `β_t = β / τ(KL_t)` (drift where you're already drifted gets penalized harder).

The cleanest paper move: **replace the global `−β KL` with the per-token weighting** and show (ablation) that the per-token version subsumes it — i.e. you can set the global `β → 0` and rely entirely on `w_t = exp(−α KL_t)`, getting equal or better results. Then ARR has *one* regularization mechanism (the per-token trust region / IS correction), not two, and `α` is the single dial. Failing that, keep both but present them as the same idea at two granularities.

**Ablation matrix for the paper:**
| config | global β | token weight `w_t` | expected |
|---|---|---|---|
| vanilla ARR | β > 0 | 1 (none) | unstable at large gap (`radpo_kl ~ 1000`) — observed |
| ARR + token weight (current) | β > 0 | `exp(−α KL_t)` | stable, +0.70 avg — observed |
| ARR, per-token only | β = 0 | `exp(−α KL_t)` | should match or beat the above — **to run** |
| ARR, per-token KL penalty too | β = 0, `β_t = β₀/τ` | `exp(−α KL_t)` | optional refinement — **to run** |

## #5 — Annealed risk operator μ (implemented)

Code: `CustomDPOConfig` now has `radpo_mu_anneal_start`, `radpo_mu_anneal_end`, `radpo_mu_anneal_frac` (all default to `None`/1.0 → no annealing → uses the fixed `radpo_confidence_level`). When `_start` and `_end` are set, `radpo_concatenated_forward` interpolates the CVaR `confidence_level` linearly from `_start` (global_step 0) to `_end` (by `_frac · max_steps`), exposes it as `self._current_radpo_mu`, and `get_batch_loss_metrics` logs it as `radpo_mu`.

Rationale (paper): early in training the student is far from the teacher, so an aggressively risk-averse `Φ^μ` operates on garbage value estimates — it should start near risk-neutral and become risk-sensitive as the student converges and the value estimates stabilize. This is a curriculum on the *nested* risk operator and fits the framework cleanly. Ablation: static-μ vs annealed-μ, same everything else.

**Convention note:** in this codebase `confidence_level` is used as `quantile(1 − confidence_level)` then `mask = value > VaR`. `confidence_level → 1` ⇒ quantile → min ⇒ mask keeps everything ⇒ ≈ expectation (risk-neutral). `confidence_level → 0` ⇒ quantile → max ⇒ mask keeps only the extreme tail ⇒ strongly risk-sensitive. So a "risk-neutral → risk-averse" schedule means `_start = 0.99` (current value, ≈ risk-neutral) → `_end` something smaller (e.g. `0.5` or `0.1`). Double-check this against the intended `Φ^μ` semantics before the first annealed run; if the convention is the other way, just swap `_start`/`_end`.

## To-do once the box is back

1. Run the **per-token-only ablation** (`beta: 0.0`, `radpo_token_weight_mode: kl_inv`, `radpo_token_weight_alpha: 1.0`, else same) — tests the #2 claim that the token weighting subsumes the global KL.
2. Run an **annealed-μ** config (e.g. `radpo_mu_anneal_start: 0.99`, `radpo_mu_anneal_end: 0.5`, `radpo_mu_anneal_frac: 1.0`) — tests #5.
3. Both alongside the queued **3-epoch token-weighted** run.
4. Eval all on the lm-eval suite; compare to the no-weight / token-weight-1ep baselines in `results/eval/SUMMARY.md`.
