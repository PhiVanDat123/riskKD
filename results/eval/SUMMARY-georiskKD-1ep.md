# GeoRiskKD 1-Epoch Sweep — Results Summary

**Date:** 2026-05-13
**Setup:** Llama 3.2-1B student ← Llama 3.1-8B DPO teacher (`vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1`), pure Ra-DPO + per-token weighting via `radpo_token_weight_mode`, 1 epoch on `pvdhihihi/ultra-feedback`, β=0.1, lr=5e-7 constant, bs=8 × 2 GPU on 2× B200, max_length=512.
**Eval:** lm-evaluation-harness via vLLM backend (TP=2), `acc_norm` where reported else `acc`, `exact_match,strict-match` for GSM8K.
**Baseline target:** `riskKD 1ep kl_inv` (token-weighted via `w_t = exp(−α·KL_t)`, α=1.0) = **43.15 avg** (per `SUMMARY.md`).

---

## Final Scoreboard (4 of 5 configs complete; 5th = risk-only was still training when the sweep was stopped)

| Method | HellaSwag (10-shot) | ARC-C (25-shot) | MMLU (5-shot) | TruthfulQA-MC2 (0-shot) | Winogrande (5-shot) | GSM8K (5-shot, strict) | **Avg** | Δ vs 43.15 |
|---|---|---|---|---|---|---|---|---|
| **riskKD 1ep no-wt** (existing baseline) | 67.74 | 38.74 | 34.38 | 49.06 | 60.85 | 3.94 | **42.45** | −0.70 |
| **riskKD 1ep kl_inv** (strongest existing baseline) | 67.82 | 41.13 | 33.78 | 48.14 | 61.80 | 6.22 | **43.15** | — |
| **georisk-risk (default mix)** | 60.95 | 38.31 | 33.43 | 48.54 | 60.22 | 0.08 | 40.26 | **−2.89** |
| **georisk-align-heavy** (λ_A=2.0) | 63.75 | 37.63 | 33.24 | 50.23 | 62.19 | 0.76 | 41.30 | **−1.85** |
| **georisk-targetall** (target=all) | 65.75 | 36.43 | 33.11 | 51.08 | 61.72 | 3.49 | **41.93** | **−1.22** ← best GeoRiskKD avg |
| **georisk-kl-conservative** (λ_D=2.0) | 62.26 | 35.32 | **34.27** | **52.75** | 61.09 | 0.15 | 40.97 | **−2.18** |
| georisk-risk-only (λ_r=1, rest=0) | (not completed — sweep stopped before finish) | | | | | | | |

## Per-task winners (vs riskKD-kl_inv = 43.15 baseline)

| Task | Best GeoRiskKD config | GeoRisk score | Baseline | Δ |
|---|---|---|---|---|
| HellaSwag | georisk-targetall | 65.75 | 67.82 | −2.07 (still loses) |
| ARC | georisk-risk | 38.31 | 41.13 | −2.82 (loses) |
| **MMLU** | **georisk-kl-conservative** | **34.27** | 33.78 | **+0.49 ✓** |
| **TruthfulQA** | **georisk-kl-conservative** | **52.75** | 48.14 | **+4.61 ✓✓** |
| Winogrande | georisk-align-heavy | 62.19 | 61.80 | **+0.39 ✓** |
| GSM8K | georisk-targetall | 3.49 | 6.22 | −2.73 (loses) |

## Bottom line

**No GeoRiskKD config beat the riskKD-kl_inv baseline on the 6-task composite at 1 epoch.** Best GeoRiskKD avg = **41.93** (targetall), still **−1.22 below** the 43.15 target.

**However**, the sweep produced two genuine per-task wins worth noting:

1. **kl-conservative wins TruthfulQA by +4.61 points** (52.75 vs 48.14). That is by far the largest single-task improvement seen from any GeoRiskKD config across the sweep. TruthfulQA measures calibration / honesty / hallucination-resistance, which is a meaningful axis for downstream safety/RAG use even when the composite suite (knowledge/reasoning-heavy) doesn't reward it.
2. **kl-conservative also nudges MMLU +0.49** (34.27 vs 33.78), the only config to beat baseline on a knowledge task.

The pattern is consistent across configs: **all GeoRiskKD variants underperform on HellaSwag / GSM8K / ARC** (completion / chain-of-thought / commonsense MC) and **partially recover or beat baseline on TruthfulQA / MMLU** (calibrated factual recall). The softmax-budget allocation appears to redistribute weight away from the fluent-continuation tokens that HellaSwag and GSM8K rely on.

## Working hypothesis for why GeoRiskKD didn't beat kl_inv at 1 epoch

1. The kl_inv baseline (`w_t = exp(−α·KL_t)`) is *strictly* a downweighting scheme — it never *increases* the weight of any token. GeoRiskKD's fixed-budget softmax (`w_t = |M_y| · softmax(q/τ)`) *redistributes* weight: high-q tokens gain mass, low-q tokens lose it. Under-trained 1-epoch runs may not yet have the policy capacity to benefit from upweighting risky tokens — they need the easy/boilerplate ones intact first to maintain fluency.
2. Per-sequence z-score normalization of features couples the geometry signal (`A_t`) to the *relative* shape of the q distribution per sequence, not absolute risk magnitude. When the sequence has few high-risk tokens, the z-score amplifies them aggressively; this is the opposite of what stability wants.
3. The 1-epoch comparison is also constrained: the baseline was tuned at 1 ep specifically. Re-running the best GeoRiskKD configs (`targetall` and `kl-conservative`) at 3 epochs would let us see if the gap narrows once token weights have time to inform the policy — historical `riskKD` 3-epoch runs only moved ±0.2 avg vs 1-epoch (per the existing `SUMMARY.md`), so this won't close a 1.22-point gap on its own. But it'd verify the per-task wins persist.

## Next steps (paused, awaiting user)

1. **Wait for risk-only to finish** (it's the simplest config; if it lands close to the others it confirms the dominant effect is the redistribution mechanism, not the lambda mix).
2. **Promote targetall + kl-conservative to 3 epochs** — those are the only two configs with a credible path to baseline parity given their per-task win patterns.
3. **Investigate the redistribution direction**: try a `w_t = exp(−q_t/τ)` variant of GeoRiskKD ("downweight high-georisk-q tokens" — same feature mix, but allocation in kl_inv's direction). If that closes the gap, the takeaway is that **z-score + softmax-budget upweighting is the wrong allocation primitive at this capacity**, even with the same features.

## Raw eval JSONs

Per-task `results*.json` files pulled from the box and saved next to this summary:

- `llama-3.2-1b-georiskKD-risk/{mmlu_full,rest}/.../results_*.json`
- `llama-3.2-1b-georiskKD-align-heavy/{mmlu_full,rest}/.../results_*.json`
- `llama-3.2-1b-georiskKD-all/{mmlu_full,rest}/.../results_*.json`
- `llama-3.2-1b-georiskKD-kl-conservative/{mmlu_full,rest}/.../results_*.json`

Trained model checkpoints stay on the box at `/workspace/riskKD/output/llama-3.2-1b-georiskKD-*/`. Per the sweep script's stage 3, each completed config also pushes to `vukien2301/llama-3.2-1b-georiskKD-<tag>` on the HF Hub.

## Existing baselines (for reference, copied from `SUMMARY.md`)

| Method | HellaSwag | ARC-C | MMLU | TruthQA | Wino | GSM8K | Avg |
|---|---|---|---|---|---|---|---|
| SFT-ep1 (8B teacher) | 81.85 | 51.19 | 63.05 | 50.59 | 78.45 | 47.61 | 62.12 |
| DPO-from-ep1 (8B teacher) | 83.05 | 52.30 | 63.39 | 57.37 | 78.22 | 52.16 | 64.42 |
| riskKD 1ep no-wt (1B) | 67.74 | 38.74 | 34.38 | 49.06 | 60.85 | 3.94 | 42.45 |
| **riskKD 1ep kl_inv (1B)** | **67.82** | **41.13** | 33.78 | 48.14 | 61.80 | **6.22** | **43.15** |
| riskKD 3ep tw — ep1 | 67.79 | 40.19 | 33.68 | 48.29 | 62.04 | 6.14 | 43.02 |
| riskKD 3ep tw — ep3 | 68.14 | 41.13 | 34.18 | 49.13 | 61.33 | 4.02 | 42.99 |
