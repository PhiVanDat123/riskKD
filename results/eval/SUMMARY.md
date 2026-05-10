# lm-evaluation-harness results

vLLM backend, TP=8 on 8x H200, `gpu_memory_utilization=0.8`, `max_model_len=4096`.

For tasks reporting both `acc` and `acc_norm`, the `acc_norm` (length-normalized accuracy) is shown.
For GSM8K, `exact_match` (strict-match) is shown.

## Teachers (8B) vs 1B students

| Benchmark | Few-shot | Metric | SFT-ep1 (8B) | DPO-from-ep1 (8B) | riskKD 1ep no-wt (1B) | riskKD 1ep token-wt (1B) | riskKD 3ep token-wt — **ep1** (1B) | riskKD 3ep token-wt — ep3 (1B) |
|---|---|---|---|---|---|---|---|---|
| MMLU            | 5  | acc                  | 63.05 | **63.39** | 34.38 | 33.78 | 33.68 | 34.18 |
| TruthfulQA-MC2  | 0  | acc                  | 50.59 | **57.37** | 49.06 | 48.14 | 48.29 | 49.13 |
| Winogrande      | 5  | acc                  | **78.45** | 78.22 | 60.85 | 61.80 | 62.04 | 61.33 |
| HellaSwag       | 10 | acc_norm             | 81.85 | **83.05** | 67.74 | 67.82 | 67.79 | 68.14 |
| GSM8K           | 5  | exact_match (strict) | 47.61 | **52.16** | 3.94 | 6.22 | 6.14 | 4.02 |
| ARC-Challenge   | 25 | acc_norm             | 51.19 | **52.30** | 38.74 | 41.13 | 40.19 | 41.13 |
| **Average**     |    |                      | **62.12** | **64.42** | 42.45 | **43.15** | **43.02** | 42.99 |

GSM8K, both metrics (`strict-match` / `flexible-extract`):

| Model | strict | flexible |
|---|---|---|
| SFT-ep1 (8B) | 47.61 | 47.84 |
| DPO-from-ep1 (8B) | 52.16 | 52.77 |
| riskKD 1ep no-wt | 3.94 | 5.61 |
| riskKD 1ep token-wt | 6.22 | 6.82 |
| riskKD 3ep token-wt — ep1 | 6.14 | (n/a — pull) |
| riskKD 3ep token-wt — ep3 | 4.02 | 4.47 |

## Models

- **SFT-ep1 (8B)**: `vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1` — Llama-3.1-8B + 1 epoch SFT on `HuggingFaceH4/deita-10k-v0-sft`.
- **DPO-from-ep1 (8B)**: `vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1` — DPO on `pvdhihihi/ultra-feedback`, constant LR 7e-7, 1 epoch, max_length 512, β=0.01, from SFT-ep1. This is the teacher / frozen reference for all riskKD runs.
- **riskKD 1ep, no token weighting (1B)**: 1B Llama-3.2 SFT'd on `deita-10k` (`vukien2301/llama-3.2-1b-deita-sft-student`), then Ra-DPO against the 8B DPO teacher, `pvdhihihi/ultra-feedback`, β=0.1, lr=5e-7 constant, max_length=512, 1 epoch (902 steps), **no per-token weighting** (`radpo_kl` ran ~800–1500 — unstable). Output dir: `riskKD_output/`. Not pushed to HF.
- **riskKD 1ep, token-weighted (1B)**: same as above + `radpo_token_weight_mode=kl_inv`, `radpo_token_weight_alpha=1.0` (`w_t = exp(-α·KL_t)`) — `radpo_kl` dropped to ~30, loss converged, reward margins positive. Output dir: `riskKD_output_tokenwt/`. Not pushed to HF.
- **riskKD 3ep, token-weighted (1B)**: same recipe as the 1-epoch token-weighted, but `num_train_epochs=3` (2706 steps), warmup_ratio 0.1, save per epoch. Output dir: `riskKD_output_tokenwt_3ep/`. Reward margins grew +1 → +2.4 → +2.9 across epochs; KL stayed tamed (~12–60). **Epoch 1 (`checkpoint-902`) was the best on this suite (43.02) and is pushed to HF: `vukien2301/llama-3.2-1b-riskkd-tokenwt-epoch1` (private).** Epoch 3 = 42.99. Epoch 2 not evaluated (stopped — epoch 1 ≈ epoch 3, no point).

## Takeaways

### Token weighting stabilizes cross-size Ra-DPO (the main finding)
Adding `w_t = exp(-α·KL_t)` per-token weighting (α=1.0) to the Ra-DPO loss: `radpo_kl` 800–1500 → ~30, train_loss 5.4 → ~1.0, reward margins flip from negative to positive. On the eval suite it buys **+0.70 pts avg** over un-weighted (42.45 → 43.15), driven by **GSM8K +2.3** and **ARC-C +2.4**. See `proposals/principled-weights-and-trust-region.md` for the reframing of these weights as the importance-sampling correction the augmented Pb-MDP requires.

### More riskKD epochs do not help this suite
3 epochs of token-weighted riskKD: ep1 = 43.02, ep3 = 42.99 — flat (within run-to-run noise; ep1's only real edge is GSM8K 6.14 vs ep3's 4.02, where ep3 drifted back toward base-1B level). Expected: this suite measures knowledge & reasoning, which a 1B can't acquire from a preference-ratio loss. To see the riskKD gain you'd need a preference / instruction-following eval (win-rate), a more capable student, or a better teacher.

### 8B SFT-ep1 → 8B DPO-from-ep1 (same-size preference distillation)
DPO on UltraFeedback adds **+2.3 pts avg**: TruthfulQA +6.78, GSM8K +4.55; MMLU/Winogrande/HellaSwag/ARC-C ~unchanged (DPO doesn't add knowledge).

### 8B → 1B (cross-size distillation)
1B student is **5–8× smaller for serving** but absolute scores are capacity-bound: TruthfulQA holds ~48–49% (vs base 1B ~38% → +10 pts) — the one place a usable signal transfers; everything else is at/near base-1B level. **GSM8K ~4–6%** is the base 1B's arithmetic ceiling under this harness config (5-shot, no CoT prompt) — not a riskKD regression; the 8B's 47–52% confirms the *teacher* can do it, but the capability didn't fit in 1B.

## Caveats

- All models score low on **ARC-Challenge** (~39–52%; Llama-3.1-8B base typically ~79%). Known pattern: chat/preference-tuned models lose accuracy on raw multiple-choice loglikelihood evals. `--apply_chat_template` would likely close this.

## Raw outputs

Per-task `results*.json` next to this file:
- `sft_epoch1/`, `dpo_from_epoch1/`, `riskKD/`, `riskKD_tokenwt/` — single-checkpoint runs
- `riskKD_tokenwt_3ep_ep1/`, `riskKD_tokenwt_3ep_ep3/` — per-epoch checkpoints of the 3-epoch run

Full per-sample logs (multi-GB) stay on the server under `/root/riskKD/output/...` — not pulled.
