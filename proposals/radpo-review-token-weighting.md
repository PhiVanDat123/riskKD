# Ra-DPO Review and Token Weighting Ideas

Last updated: 2026-05-11

## Sources

- Ra-DPO OpenReview: https://openreview.net/forum?id=V4oTkK7cQz
- Hugging Face paper page: https://huggingface.co/papers/2505.20359

## Paper Summary

**Risk-aware Direct Preference Optimization under Nested Risk Measure** proposes Ra-DPO, a risk-aware extension of DPO. The paper argues that standard DPO and related preference optimization methods mainly control policy drift through expectation-style KL regularization, while LLM generation is sequential and token-level. Ra-DPO therefore introduces a **sequential risk ratio** based on nested risk measures, such as CVaR or entropic risk, to control token-level deviation from a reference model.

The objective tries to balance two goals:

1. Improve preference alignment, as in DPO.
2. Suppress risky drift from the reference policy through token-level risk control.

The paper evaluates on IMDb, Anthropic HH, and AlpacaEval, and reports better tradeoffs between reward accuracy / win rate and model drift.

## Strengths

- The paper identifies a real weakness of sequence-level DPO: preference optimization happens over full responses, but generation is token-by-token.
- The nested-risk framing gives a principled way to focus on tail behavior rather than only mean behavior.
- The method is compatible with offline preference data and does not require online rollouts.
- The sequential risk ratio gives useful diagnostics, especially token-level KL / drift curves.

## Weaknesses

- The paper mostly defines "risk" as **drift from the reference model**. That is useful, but narrow. Drift is not the same as semantic risk such as toxicity, hallucination, unsafe refusal, or low truthfulness.
- If the reference model is weak, biased, or miscalibrated, staying close to it may preserve bad behavior.
- In a large teacher to small student setting, high KL can reflect capacity mismatch rather than true behavioral risk.
- The method may improve training metrics without improving downstream benchmarks if the evaluation suite measures knowledge/reasoning rather than alignment quality.
- CVaR implementation details matter a lot: the confidence-level convention, tail normalization, and whether the risk is computed over vocabulary tokens or sequence positions can change the meaning of the objective.

## Relevance to riskKD

In this repo, the hard case is **8B teacher -> 1B student**. The observed failure mode was:

- Raw Ra-DPO / ARR had very large teacher-student KL.
- Reward margins flipped sign or stayed noisy.
- Reward accuracy stayed near random.
- Token KL weighting made training numerically stable, but benchmark gains were modest.

This suggests that token weights should be treated as **trust weights**:

> Trust teacher risk/value guidance only where the student and teacher distributions overlap enough for that guidance to be meaningful.

That is different from saying high KL always means semantic risk. In cross-size distillation, high KL often means the 1B student cannot represent the 8B teacher's local logit landscape.

## Token Weighting Design Goals

A good token weight should:

- Reduce high-variance teacher-student mismatch.
- Preserve enough preference signal to improve the student.
- Avoid shrinking the entire loss just because many tokens get small weights.
- Be easy to log and ablate.
- Distinguish true risk from teacher-student capacity mismatch where possible.

Always log:

```text
radpo_raw_kl/*
radpo_kl/*                 # weighted KL
radpo_token_weight/*
radpo_effective_tokens/*
radpo_rewards/margins
radpo_rewards/accuracies
```

## Recommended Token Weighting Methods

### 1. Normalized KL-Inverse Weight

Formula:

```text
w_t = exp(-alpha * KL_t)
w_t <- w_t * valid_token_count / sum_t w_t
```

Why it may work:

- High-KL tokens are where teacher and student disagree strongly.
- In the 8B -> 1B case, those positions are often noisy for distillation.
- Normalization prevents the loss from improving only because total weight shrinks.

Use this as the default baseline.

Recommended sweep:

```text
alpha in {0.25, 0.5, 1.0, 2.0}
```

### 2. Risk-Only KL Weighting

Formula:

```text
u     = sum_t beta * log(pi_theta / pi_ref)
delta = sum_t w_t * risk_t
loss  = -log sigmoid(u - delta)
```

Why it may work:

- The preference utility gap still learns from the pair label.
- Only the risk correction is damped.
- This is conceptually clean if the risk term is the unstable part.

Risk:

- In 8B -> 1B distillation, the utility gap `u` also contains teacher-student log-ratios and can be noisy.
- This may be less stable than all-term weighting.

### 3. All-Term KL Weighting

Formula:

```text
u     = sum_t w_t * beta * log(pi_theta / pi_ref)
delta = sum_t w_t * risk_t
loss  = -log sigmoid(u - delta)
```

Why it may work:

- More faithful to the importance-sampling / trust-region story.
- Both the utility gap and risk correction are unreliable at low-overlap states.
- Likely stronger in cross-size transfer.

This should be run as a direct ablation against risk-only weighting.

### 4. Clipped KL-Inverse Weight

Formula:

```text
w_t = clamp(exp(-alpha * KL_t), w_min, w_max)
w_t <- normalized(w_t)
```

Suggested values:

```text
w_min = 0.05 or 0.1
w_max = 2.0
```

Why it may work:

- Prevents informative high-KL tokens from being completely ignored.
- Prevents very low-KL tokens from dominating after normalization.
- More stable than pure exponential weighting.

### 5. Teacher-Confidence Weight

Formula:

```text
H_t = entropy(pi_ref(. | s_t))
w_t = exp(-alpha * KL_t) * sigmoid(gamma * (H0 - H_t))
```

Why it may work:

- Teacher guidance is more useful when the teacher is confident.
- High teacher entropy means the teacher itself is uncertain.
- Combines student-teacher overlap with teacher reliability.

Risk:

- Very low entropy can mean overconfidence.
- Use clipping or a bell-shaped confidence gate if needed.

Alternative bell-shaped version:

```text
w_conf_t = exp(-eta * (H_t - H_target)^2)
w_t = exp(-alpha * KL_t) * w_conf_t
```

### 6. Teacher-Margin Weight

For chosen/rejected responses:

```text
m_t = log pi_ref(y_chosen_t | s_t) - log pi_ref(y_rejected_t | s_t)
w_t = sigmoid(gamma * m_t)
```

Combined with KL:

```text
w_t = exp(-alpha * KL_t) * sigmoid(gamma * m_t)
```

Why it may work:

- Many tokens are preference-neutral.
- Teacher-margin weighting focuses learning on positions that explain the chosen/rejected distinction.
- Helps avoid wasting weight on boilerplate or shared prefixes.

Implementation note:

- Token alignment between chosen and rejected sequences can be messy because lengths differ.
- A simpler first version can use sequence-level teacher margin as a sample weight.

### 7. CVaR Soft-Tail Weight

Instead of a hard mask:

```text
mask_t = 1[risk_t > VaR_mu]
```

use:

```text
w_t = sigmoid(lambda * (risk_t - VaR_mu))
```

Why it may work:

- Hard CVaR masks are discontinuous and can be noisy.
- Soft tail weights keep near-tail tokens in the gradient.
- This may stabilize training while preserving risk sensitivity.

Recommended sweep:

```text
lambda in {1, 2, 5, 10}
```

### 8. Position-Aware Weight

Formula:

```text
w_t = exp(-alpha * KL_t) * gamma^(T - t)
```

with `gamma < 1` to emphasize later response tokens.

Why it may work:

- Early tokens often include chat boilerplate or generic setup.
- Later tokens often carry actual answer content, reasoning, final answer, or refusal.
- Can be useful for instruction-following datasets.

Risk:

- Some tasks encode crucial safety behavior early.
- Use this only as an ablation, not the default.

### 9. Prompt/Response Masked Weighting

Apply token weights only to assistant response tokens, not prompt/template tokens.

Why it may work:

- Preference optimization is about response behavior.
- Prompt/chat-template tokens can dominate length but carry no trainable preference signal.

This should be standard if the current data collator exposes response masks clearly.

### 10. Semantic Risk Weight

Use an external reward model or safety model:

```text
risk_t = low truthfulness / high toxicity / high refusal error / reward-model tail score
w_t = CVaR_or_soft_tail(risk_t)
```

Why it may work:

- It addresses the biggest conceptual weakness of Ra-DPO: risk as mere reference drift.
- It can target actual bad-tail behavior.
- It gives stronger paper claims if evaluated on safety/truthfulness tails.

Risk:

- Requires extra scoring or generated samples.
- Reward model bias can leak into the method.

## Recommended Run Matrix

Start with these four:

| Config | Weight target | Normalize | Extra weighting | CVaR |
|---|---:|---:|---|---|
| A | `risk` | yes | none | `mu: 0.99 -> 0.5` |
| B | `all` | yes | none | `mu: 0.99 -> 0.5` |
| C | `all` | yes | clipped KL | `mu: 0.99 -> 0.5` |
| D | `all` | yes | teacher margin | `mu: 0.99 -> 0.5` |

Suggested interpretation:

- If A is stable but weak: risk correction was noisy, but utility gap still lacks useful signal.
- If B beats A: cross-size noise affects both utility and risk terms.
- If C beats B: pure exponential weighting is too aggressive.
- If D beats C: preference-relevance matters more than raw distributional overlap.

## My Current Bet

For 8B teacher -> 1B student, the most likely winner is:

```text
w_t = normalized(clamp(exp(-alpha * KL_t), 0.05, 2.0))
target = all
```

Then improve it with teacher-margin weighting:

```text
w_t = normalized(
    clamp(exp(-alpha * KL_t), 0.05, 2.0)
    * sigmoid(gamma * teacher_margin_t)
)
```

Reason:

- The utility gap and risk correction both contain teacher-student log-ratio terms.
- Large cross-size KL corrupts both, not only the risk term.
- Clipping prevents the method from throwing away too many hard but informative tokens.
- Teacher margin focuses the remaining signal on preference-relevant positions.

## Evaluation Advice

Do not judge token weighting only by train loss or weighted KL.

Required diagnostics:

```text
raw KL decreases or stays bounded
weighted KL decreases
mean token weight does not collapse
effective token count remains reasonable
reward margins improve without exploding
reward accuracy rises above 0.5
eval scores do not regress
```

Also add at least one alignment-oriented evaluation:

- AlpacaEval 2
- Arena-Hard
- MT-Bench
- TruthfulQA generation
- safety/toxicity tail benchmark
- reward-model worst-decile score

The current lm-eval suite mostly measures knowledge and reasoning. A 1B model is unlikely to gain much knowledge from a preference-ratio loss, so a flat lm-eval result does not necessarily mean the risk objective failed.

## Implementation Checklist

- Keep the explicit teacher/reference model for Ra-DPO even when `dpo_weight = 0`.
- Log raw KL separately from weighted KL.
- Normalize token weights per sequence.
- Add `target = risk` vs `target = all` as a YAML ablation.
- Consider clipping KL weights.
- Replace hard CVaR masks with optional soft-tail masks.
- Avoid `softmax().log()` underflow; use `log_softmax()`.
- Revisit `is_split_risk_ratio`; full-vocab CVaR is easier to justify than half-vocab splitting.
