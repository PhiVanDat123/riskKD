# Token-weighting schemes for riskKD / Ra-DPO

A menu of ways to compute the per-token weight `w_t` used inside `radpo_concatenated_forward` →
`_radpo_get_batch_logps`. Grouped by the signal each one keys off. The Ra-DPO / CVaR-native
options (Family E) are the ones most directly tied to the risk story; the rest are listed for
contrast and ablation.

Notation: `s_t` = state (prompt + tokens so far), `y_t` = the token at position t,
`π_θ` = student, `π_ref` = reference (the DPO-trained teacher in our setup),
`π_tch` = teacher token distribution, `H(·)` = entropy, `T` = number of valid (loss-masked) tokens
in the sequence, `μ` = CVaR confidence level (`radpo_confidence_level`).

---

## A. What we have now (baseline)

**`kl_inv`** — `w_t = exp(−α · KL(π_θ(·|s_t) ‖ π_ref(·|s_t)))`.

This is a **trust-region** weight: it *down-weights* tokens where the student has already drifted
from the teacher/ref. Note this is philosophically the *opposite* of a risk weight — a risk weight
*up-weights* the bad tail. If you ever combine the two, multiply them deliberately:
`w_t = w_t^{trust-region} · w_t^{risk}`.

---

## B. Teacher-confidence weights (cheap, no extra model)

1. **Teacher entropy** — `w_t ∝ exp(−α · H(π_tch(·|s_t)))`. Trust decisive teacher positions.
   (Use the inverse if you instead want the student to spend capacity on the ambiguous ones.)
2. **Teacher peakedness** — `w_t = π_tch(top1 | s_t)`, or the top1−top2 probability margin.
3. **Agreement gate** — `w_t = 1[argmax π_tch(·|s_t) == y_t]`. Only distill where the teacher
   would actually have produced the chosen/rejected token.

---

## C. Divergence-tempered importance-sampling weights (trust-region family)

4. **Per-token log-ratio temper** — `w_t = exp(−α · |log π_θ(y_t|s_t) − log π_ref(y_t|s_t)|)`.
   Token-level PPO-style ratio damping. Directly attacks the 8B→1B blow-up
   (`logps/chosen ≈ −540` vs `logps/rejected ≈ −1064`).
5. **Clipped IS** — `w_t = clip(π_θ(y_t|s_t)/π_ref(y_t|s_t), 1−ε, 1+ε)` per token (hard PPO clip
   instead of the soft exp).
6. **Swap the divergence** — Jensen–Shannon, total-variation, or χ² instead of KL inside the
   exponent. Bounded → less explosive than KL on a large teacher.

---

## D. Value / advantage weights (TVKD-native)

7. **Soft-value advantage** — `w_t ∝ exp(β · A_t)` with `A_t = V(s_{t+1}) − V(s_t)` from the
   `shifted_v_fn` soft-value already computed for the Q-adapter term. Tokens that move the
   teacher's value the most get the weight. Most "in-family" option — the machinery already exists.
8. **Implicit token reward (Rafailov et al., "From r to Q\*")** — `w_t ∝ |β · log π_θ(y_t|s_t)/π_ref(y_t|s_t)|`,
   the DPO implicit per-token reward magnitude.

---

## E. Risk-aware / CVaR-native weights — the Ra-DPO connection

Key fact: `CVaR_μ(L) = E[L | L ≥ VaR_μ(L)]`, and **its subgradient is exactly a token weight**:
`w_t = 1[ℓ_t ≥ VaR_μ] / (1−μ)`. So "Ra-DPO token weighting" *is* "weight tokens by tail-membership
of their per-token loss contribution `ℓ_t`." Variants, ordered hard → smooth:

9. **Hard tail-membership (textbook CVaR)** — per token compute the loss contribution `ℓ_t`
   (e.g. the negative log-ratio term, or the distillation KL); find the empirical (1−μ)-quantile
   across tokens in the sequence (or batch); set `w_t = 1/(1−μ)` for tokens above it, `0` below.
   Faithful to the CVaR definition. Downsides: hard indicator → noisy gradients; quantile recompute
   every step.
10. **Rockafellar–Uryasev with a learned threshold** — same as #9 but instead of an empirical
    quantile, learn the VaR scalar `η` online via the RU objective
    `min_η  η + 1/(1−μ) · E[(ℓ_t − η)_+]`; weight `w_t = 1/(1−μ) · 1[ℓ_t > η]`.
    One extra scalar parameter, much more stable than recomputing quantiles — this is how CVaR is
    normally optimized in deep RL.
11. **Soft-CVaR / entropic-risk weights** ← *recommended* — `w_t ∝ exp(λ · ℓ_t)` (a softmax over
    per-token loss contributions, temperature `1/λ`). This is the gradient of the entropic risk
    `1/λ · log E[exp(λ ℓ)]`, which is exactly KL-ball DRO and a smooth surrogate for CVaR.
    Fully differentiable, no quantile, no indicator; degrades to plain expectation as `λ → 0` — so it
    composes naturally with the μ-annealing schedule already in the code (anneal `λ` alongside `μ`).
    Easiest to drop in, least likely to add training noise.
12. **Spectral / distortion risk weights** — `w_t = φ(rank_t / T)` for a non-increasing spectrum `φ`
    (CVaR = uniform on the top-(1−μ) slice; could use exponential or power spectra), or the
    Wang-transform `w_t = g'(F̂(ℓ_t))`. Generalizes CVaR; tune how hard the tail dominates without a
    sharp cutoff.
13. **χ²-DRO / variance-emphasis** — `w_t ∝ (1 + λ(ℓ_t − mean ℓ))_+`. First-order tail emphasis,
    ~free, the linearization of CVaR; a cheap ablation point between "uniform" and "full CVaR."

---

## F. Risk applied to *what* — design choices that matter as much as the formula

- **Risk on the preference margin** — `ℓ_t = per-token contribution to (chosen − rejected) log-ratio`.
  Up-weighting the tail → robust to the *hardest* tokens (true CVaR-DPO). But if preference labels
  are noisy you may want the *opposite*: trim the tail (Huber / trimmed-mean weights) so a few
  mislabeled tokens don't dominate.
- **Risk on the distillation residual** — `ℓ_t = KL(π_tch(·|s_t) ‖ π_θ(·|s_t))`. CVaR over this =
  "spend the student's limited capacity on the tokens it imitates *worst*" — a natural KD objective
  and arguably the cleanest story for *riskKD* specifically (risk-aware *distillation*, not just
  risk-aware preference).

---

## Suggested path

For Ra-DPO faithfulness with minimal training instability:

1. **#11 (soft-CVaR, `w_t ∝ exp(λ ℓ_t)`)** applied to the **risk-correction term only**
   (matches the current `radpo_token_weight_target: risk`), with `λ` annealed on the same schedule
   as `μ`.
2. Keep **#10 (RU learned-η hard CVaR)** as the "strict" ablation.
3. Keep **#9** as the sanity check that #10 and #11 agree in the limit.
4. Log raw vs weighted loss for all three (the `radpo_raw_kl/*`, `radpo_token_weight/*`,
   `radpo_effective_tokens/*` metrics already do this).

### Sketch for #11 inside `_radpo_get_batch_logps`

~10 lines: after computing the per-token loss contribution `l_t` (shape `[B, T]`, already masked),

```python
# soft-CVaR / entropic-risk token weights
lam = self.args.radpo_softcvar_lambda          # new YAML knob; lam -> 0 recovers expectation
l_t = l_t.detach()                              # weights are a (stop-grad) reweighting, not a 2nd loss
l_t = l_t.masked_fill(~loss_mask, float("-inf"))
w_t = torch.softmax(lam * l_t, dim=-1)          # sums to 1 over valid tokens per sequence
w_t = w_t * loss_mask.sum(-1, keepdim=True)     # rescale to mean-weight 1  (same as radpo_token_weight_normalize)
```

then feed `w_t` through the existing `radpo_token_weight_target` / `radpo_token_weight_normalize`
plumbing instead of the `kl_inv` weights. Anneal `lam` from `0` (or a small value) up to its target
on the same step schedule as the μ anneal so early training is near-risk-neutral.

---

# Novel directions (research-grade)

The schemes above are the "known toolbox." Below are ideas that are (to the best of my knowledge)
*not* in the DPO / KD / risk-RL literature in this exact form, ranked by how naturally they fit
riskKD = TVKD + Ra-DPO with an 8B teacher → 1B student. The capacity gap and the time-inconsistency
of static CVaR are the two structural facts that make these worth pursuing.

### N1. Nested / recursive CVaR soft-values — token weight as the risk-tilted change of measure *(flagship; deep dive below)*

Ra-DPO applies a **single static** CVaR over the whole trajectory's log-ratio, which is
**time-inconsistent** (Ruszczyński 2010; Shapiro). Replace it with a **nested** risk operator inside
the TVKD soft-value: `V_t(s) = CVaR_{μ, a∼π_ref(·|s)}[ r(s,a) + γ V_{t+1}(s') ]`, applied
recursively. This is time-consistent, and the per-token weight is the Radon–Nikodym derivative the
inner CVaR induces at each step: `w(a|s) = (1/μ)·1[Q(s,a) in the bad μ-tail]`. TVKD already computes a
soft value with `qadapter_softvalue_type ∈ {sum, entropy, …}` (= aggregator over the teacher's
action distribution); this is **one new aggregator option**, `"cvar"`, plus a dual variable. The
static-vs-nested distinction is *literally* the fix for "CVaR-DPO weights the tail of the whole
sequence, not the bad tokens." Full write-up below.

### N2. Preferential information content — weight a token by how much it carries the preference label

In a chosen/rejected pair, most tokens are identical filler; the preference signal lives at a few
positions. Since the precomputed teacher logits give `π_tch(·|s_t)` for **both** branches, estimate
the per-token mutual information with the preference label:
`w_t ∝ I(y_t ; chosen-vs-rejected | s_t) ≈ KL(π_tch^{chosen}(·|s_t) ‖ π_tch^{rejected}(·|s_t))`
(or the JS divergence between the branch-conditioned teacher distributions). Up-weight exactly the
tokens where the teacher's behavior diverges between the good and bad continuation; ~zero elsewhere.
This is a teacher-derived **credit assignment** for "which tokens does this preference judgement
depend on" — the under-addressed weakness of token-level DPO. Composes with N1 (use it as the
per-step reward `r_t`).

### N3. Rate–distortion / capacity-aware water-filling weights

8B→1B distillation is hard *because the 1B student has a fixed bit budget*. Model it directly:
maximize `−Σ_t w_t · KL(π_tch ‖ π_θ)` s.t. `Σ_t (info the student must store at t) ≤ C`. The
Lagrangian's optimal weights satisfy a **water-filling** condition `w_t ∝ max(0, ν − 1/sensitivity_t)`:
cheap tokens (low teacher entropy) get a flat ration, expensive tokens get budget only if worth it,
hopeless ones get cut. Proxy for `sensitivity_t`: teacher entropy, or `‖∂ℓ_t/∂(student last-layer
features)‖`. Only scheme here that names the *actual* bottleneck as the constraint; gives an
interpretable "the student gave up on these N% of tokens" diagnostic.

### N4. CVaR token weighting ≡ gradient-variance reduction *(a theorem, not just a method)*

Variance-optimal SGD importance weights are `w_t ∝ ‖∇ℓ_t‖` (Katharopoulos & Fleuret; Alain et al.) —
known at the *sample* level, unused at the *token* level for DPO. The interesting bit: in 8B→1B the
per-token gradient norm is dominated by the few tokens where the student is catastrophically wrong —
which is *also* the CVaR tail. So `‖∇ℓ_t‖`-weighting and CVaR-weighting coincide to leading order;
under a local-quadratic assumption you can likely *prove* the equivalence and present "risk-aware
token weighting" and "variance-reduced token distillation" as two views of one object — giving Ra-DPO
a free-lunch *optimization* justification, not just a robustness one. Practical form:
`w_t ∝ exp(λ·‖∇_h ℓ_t‖)` with an EMA on the norms.

### N5. Doubly-robust / control-variate token-DPO weights

The DPO log-ratio is a high-variance IS estimator; with an 8B ref it explodes (the
`logps/chosen ≈ −540` vs `−1064` symptom). Use the teacher distribution as a **control variate**:
`ℓ_t^{DR} = ℓ_t − (π_θ(y_t|s_t)/π_ref(y_t|s_t) − 1)·b_t` with baseline `b_t` = expected reward under
π_ref (TVKD's soft value already provides it). This is the doubly-robust off-policy estimator
(Dudík; Jiang & Li) ported to token-level DPO — the control variate cancels the leading-order ratio
term, so variance is bounded *by construction* instead of by an `exp(−α·KL)` band-aid. The most
principled fix for the ratio-explosion you actually observed.

### N6. Distributionally-robust over the *teacher's branching*, not over the data

Vanilla Ra-DPO takes CVaR over the dataset. In distillation you also have `π_tch(·|s_t)` — so take
the risk over **where the teacher might go**: `w_t ∝ CVaR_{μ, y∼π_tch(·|s_t)}[ ℓ(y | s_t) ]`. Force
the student to match the teacher *especially on the teacher's own high-uncertainty / multimodal
positions* — a calibration objective, orthogonal to data-level risk, stackable with it. Risk over the
*model's* aleatoric branching as a distillation weight.

### N7. Epistemic risk over a reward-model ensemble

With a small reward-head ensemble, weight tokens by **disagreement among heads** about that token's
marginal contribution to the preference. Puts capacity where the preference signal is *epistemically*
shaky → robustness to reward misspecification (a known DPO failure mode). "CVaR over the reward
posterior" rather than "CVaR over the data" — a more defensible robustness claim than vanilla Ra-DPO's,
and a clean ablation pair (aleatoric vs epistemic risk weighting).

### Bonus — a guarantee, not a method: conformal risk control on the threshold

Pick the VaR threshold `η` (equivalently the support of `w`) by **conformal risk control**
(Angelopoulos et al.) so the up-weighted token set provably captures a target fraction `1−μ` of
validation preference-violation mass. Turns "risk-aware" from a vibe into a distribution-free
certificate on post-distillation preference accuracy.

### Filed under "exotic" (know they exist, probably not worth it here)

- **Shapley-value token weights** — fair attribution of the sequence preference to tokens;
  exponential cost, only linear-time approximations, and those assume near-additivity.
- **Optimal-transport token weights** — an OT plan between teacher and student token embeddings, with
  the student-side marginal mass = the weight. Real idea for *cross-tokenizer* distillation; overkill
  for Llama→Llama with a shared tokenizer.
- **Leave-one-token-out influence weights** — `w_t ∝ |Δ(val preference acc) when token t's loss is
  removed|`, via a first-order influence approximation (one extra HVP-free backward). Token-level data
  attribution; the exact version is hopeless, the approx is doable but noisy.

---

# Deep dive: nested-CVaR soft-values (idea N1)

## The defect in static Ra-DPO

Ra-DPO replaces the expectation in the DPO objective with `CVaR_μ` over the per-example log-ratio
`L`. As a *static* risk measure applied once to the whole sequence, its subgradient is a single
sequence-level indicator: `w(τ) = (1/μ)·1[L(τ) ≥ VaR_μ]`. Consequences:

1. **It can't localize.** Every token of a "tail" sequence gets the *same* weight `1/μ`; every token
   of a non-tail sequence gets `0`. But a 1B student doesn't fail at *whole sequences* — it fails at
   *specific positions* (a hard reasoning step, a rare entity, a long-range dependency). Static CVaR
   has no way to express "this token, in an otherwise-fine sequence, is the problem."
2. **It is time-inconsistent.** A dynamic risk assessment `{ρ_t}` is *time-consistent* iff
   `ρ_{t+1}(X) ≤ ρ_{t+1}(Y) ⇒ ρ_t(X) ≤ ρ_t(Y)` — "if Y looks riskier than X from tomorrow's
   vantage point, it looks riskier from today's." Static CVaR violates this (classic counterexample:
   two scenarios that swap which is worse depending on the conditioning step). The practical effect:
   the implied "what should the policy do" can *flip mid-sequence* — the objective rewards one kind of
   behavior in the first half of a response and a contradictory kind in the second. That shows up as
   training instability and incoherent generations, and it's exactly the kind of pathology that's
   easy to hand-wave past in the loss math but bites in practice.

## The fix: a nested risk MDP

Treat token generation as an MDP: state `s_t = (prompt, y_{<t})`, action `a = y_t`, the teacher's
DPO-trained policy `π_ref(·|s_t)` as the action distribution, a per-token reward `r(s_t, a)` (e.g.
the implicit DPO token reward `β·log π_θ/π_ref`, or the margin term, or N2's preferential info — any
of these). Define the value by a **nested** Bellman recursion with a coherent one-step risk operator
`ρ`:

```
V_t(s) = ρ_{ a ∼ π_ref(·|s) } [ r(s, a) + γ · V_{t+1}(s') ],     V_T ≡ 0
```

with `ρ = CVaR_μ`. Because the risk is re-applied at *every* step to the *one-step-ahead*
distribution (and `V_{t+1}` already has all downstream risk baked in recursively), this construction
is **time-consistent** — in fact the nested-CVaR is essentially *the* law-invariant time-consistent
dynamic risk measure you get from CVaR (Ruszczyński 2010; Shapiro 2009; Bäuerle & Ott 2011 for the
MDP/CVaR specifics). This is also literally what the riskKD paper means by "nested CVaR risk
operator" — the static implementation in the current code is a simplification of it.

## What the token weight becomes

Use the Rockafellar–Uryasev dual at each state. For a *value/reward* `Q(s,·) = r(s,·) + γ V_{t+1}(·)`
where the bad tail is the *low* tail:

```
CVaR_μ[Q(s,·)]  =  max_η  { η − (1/μ) · E_{a∼π_ref(·|s)}[ (η − Q(s,a))_+ ] }
```

at the optimum `η*(s) = VaR_μ(Q(s,·))`, and the gradient of the inner expectation w.r.t. the action
distribution is supported on the tail. So the inner CVaR is equivalent to evaluating an ordinary
expectation under a **risk-tilted** action distribution `π̃(a|s) = π_ref(a|s) · ξ(a|s)` with

```
ξ(a|s)  =  (1/μ) · 1[ Q(s,a) ≤ η*(s) ]          # the Radon–Nikodym derivative / change of measure
```

`ξ` is exactly a **per-token weight on the distillation/log-ratio term**. And crucially, in the
*nested* formulation the weight at step t is just the *local* `ξ(y_t | s_t)` — there's no product over
the trajectory, because each `V_{t+1}` already carries the downstream risk. Contrast:

| | static Ra-DPO (current) | nested CVaR (proposed) |
|---|---|---|
| token weight | `w_t = (1/μ)·1[ L(whole seq) ≥ VaR_μ ]` — same for all tokens of a tail seq, 0 otherwise | `w_t = (1/μ)·1[ Q(s_t, y_t) in bad μ-tail of π_ref(·|s_t) ]` — per position |
| localizes to bad tokens? | no | **yes** |
| time-consistent? | no | **yes** |
| extra parameters | 0 (uses an empirical sequence quantile) | a per-state VaR `η(s)` — either an empirical vocab quantile (0 params) or a learned scalar/head |
| recovers TVKD at μ→1 | n/a (it's a DPO term) | **yes, exactly** (`ξ ≡ 1`) — clean sanity check & anneal target |

## Implementation

1. **New knob.** `qadapter_softvalue_type: "cvar"` in `CustomDPOConfig`; reuse `radpo_confidence_level`
   for μ (or add `qadapter_cvar_mu`).
2. **Swap the aggregator in `shifted_v_fn`.** Wherever the soft value is currently `logsumexp` over
   `Q(s,·)` weighted by `π_ref(·|s)` (the `entropy` type) or a plain weighted mean (`sum`), replace it
   with the RU-CVaR estimate. You need `VaR_μ(Q(s,·))` per token. Two options:
   - **(a) exact empirical quantile over the vocab** — form the weighted CDF of `Q(s,a)` under
     `π_ref(a|s)` (sort `Q` along the vocab axis, cumsum the sorted probs, find the crossing of μ).
     `O(V log V)` per token; vectorizes over `[B, T]`; `V ≈ 128k` is fine on H200, and you only need a
     partial sort. **Parameter-free — start here.**
   - **(b) learned dual** — `η(s)` = a scalar (or a tiny linear head off the hidden state) trained by
     the RU gradient `∂/∂η [ η − (1/μ)(η − Q)_+ ]`. One extra parameter, smoother, the standard move
     in risk-sensitive RL — switch to this only if (a) is too slow or the quantile flickers.
3. **Token weight → existing plumbing.** `ξ(y_t|s_t) = (1/μ)·1[Q(s_t,y_t) ≤ η(s_t)]` multiplies the
   per-token distillation / log-ratio term; pass it through the existing
   `radpo_token_weight_normalize` (per-sequence renorm to mean-weight 1) and `radpo_token_weight_target`
   machinery so you don't re-derive that.
4. **Anneal μ from 1 → target** on the schedule that already exists. At μ=1, `ξ ≡ 1` and a run must
   match a plain TVKD run *bit for bit* — that's the regression test that the operator swap didn't
   change anything else.
5. **Log** `radpo_raw_kl/*` (unweighted), `radpo_token_weight/*` (mean ξ), `radpo_effective_tokens/*`
   (count of tail tokens ≈ `μ · T`) — the metrics you already added cover this; just point them at ξ.

## Smooth variant (almost free)

The hard `1[Q ≤ η]` is the source of gradient noise. Replace it with a Boltzmann tilt toward low `Q`:
`ξ(a|s) ∝ exp( −Q(s,a) / τ )`. But that's *exactly* the existing `entropy` soft-value with a
**negative temperature**: positive τ tilts toward *high* Q (optimistic / KL-regularized RL), negative
τ tilts toward *low* Q (pessimistic / risk-averse). As `τ → 0⁻` it → hard CVaR_{μ→0} (worst-case);
as `τ → −∞` it → uniform (expectation). So the entire expectation ↔ worst-case family is reachable by
**allowing `qadapter_temperature < 0`** — zero new estimator code, just lift a sign restriction, and
you get a differentiable surrogate for the μ-family of CVaRs (anneal τ from a large negative value
toward `0⁻`).

## Why this is the pick

- **Most novel** — nested/recursive risk measures are standard in risk-sensitive RL, but porting one
  into a *distillation soft-value* and reading the *token weight* off the change of measure is, as far
  as I know, new; and it gives the riskKD paper's "nested CVaR" framing an actual implementation.
- **Smallest change per unit of conceptual weight** — one aggregator option + one dual variable; the
  token-weight plumbing, the μ-anneal, and the metrics already exist.
- **Fixes a real defect**, not just adds a knob — static Ra-DPO is genuinely time-inconsistent and
  genuinely can't localize; this addresses both, with a clean μ→1 ⇒ TVKD limit for sanity.
- **Subsumes the soft-CVaR weight (#11)** as the τ<0 Boltzmann special case, and pairs naturally with
  N2 (preferential information) as the choice of per-step reward `r_t`.

### References (for the writeup, not exhaustive)

- Rockafellar & Uryasev 2000 — CVaR optimization (the dual / `η` trick).
- Ruszczyński 2010, *Risk-averse dynamic programming for Markov decision processes* — nested/Markov
  risk measures, time-consistency.
- Shapiro, Dentcheva & Ruszczyński 2009/2014, *Lectures on Stochastic Programming* — coherent &
  dynamic risk measures.
- Bäuerle & Ott 2011 — MDPs with CVaR; Bäuerle & Rieder for risk-sensitive control more broadly.
- Tamar, Glassner & Mannor 2015 — policy gradient for CVaR; Chow, Tamar, Mannor & Pavone 2015 —
  risk-constrained RL with CVaR.
- Du, Wang et al. 2023 — *Risk-sensitive RL with iterated CVaR* (the iterated/nested-CVaR operator in
  the RL-theory setting).
- Rafailov et al. 2024, *From r to Q\**: the token-level MDP / implicit-reward view of DPO (for the
  choice of per-token reward `r_t`).
