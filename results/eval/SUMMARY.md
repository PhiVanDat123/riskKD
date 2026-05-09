# lm-evaluation-harness results

vLLM backend, TP=8 on 8x H200, `gpu_memory_utilization=0.8`, `max_model_len=4096`.

For tasks reporting both `acc` and `acc_norm`, the `acc_norm` (length-normalized accuracy) is shown.
For GSM8K, `exact_match` (strict-match) is shown.

## Side-by-side

| Benchmark | Few-shot | Metric | SFT-epoch1 | DPO-from-epoch1 | Δ (DPO − SFT) |
|---|---|---|---|---|---|
| MMLU                | 5  | acc                          | 63.05% | **63.39%** | +0.34 |
| TruthfulQA-MC2      | 0  | acc                          | 50.59% | **57.37%** | **+6.78** |
| Winogrande          | 5  | acc                          | **78.45%** | 78.22% | -0.23 |
| HellaSwag           | 10 | acc_norm                     | 81.85% | **83.05%** | +1.20 |
| GSM8K               | 5  | exact_match (strict)         | 47.61% | **52.16%** | **+4.55** |
| ARC-Challenge       | 25 | acc_norm                     | 51.19% | **52.30%** | +1.11 |
| **Average**         |    |                              | **62.12%** | **64.42%** | **+2.30** |

## Models

- **SFT-epoch1**: `vukien2301/llama-3.1-8b-deita-sft-teacher-epoch1` — Llama-3.1-8B + 1 epoch SFT on `HuggingFaceH4/deita-10k-v0-sft`.
- **DPO-from-epoch1**: `vukien2301/llama-3.1-8b-ultrafeedback-dpo-from-epoch1` — DPO on `pvdhihihi/ultra-feedback`, constant LR 7e-7, 1 epoch, max_length 512, β=0.01, starting from SFT-epoch1.

## Reading the gap

DPO on UltraFeedback **adds +2.3 pts on average** across the six benchmarks vs the pre-DPO checkpoint:
- **TruthfulQA +6.78**: largest single jump — UltraFeedback alignment data directly helps the model prefer truthful completions.
- **GSM8K +4.55**: chain-of-thought reasoning improved (DPO data includes reasoning preferences).
- MMLU / Winogrande / HellaSwag / ARC-C: mostly unchanged — DPO doesn't add factual or commonsense knowledge, just preference shaping.

## Caveats

- Both models score unusually low on **ARC-Challenge** (~51-52%; Llama-3.1-8B base typically ~79%). This is a known regression pattern: chat/preference-tuned models lose accuracy on raw multiple-choice loglikelihood evals because they have learned to prefer the chat format. Re-evaluating with `--apply_chat_template` would likely close this gap.

## Raw outputs

Each per-task `results*.json` lives next to this file:
- `dpo_from_epoch1/{mmlu,truthfulqa_mc2,winogrande,hellaswag,gsm8k,arc_challenge}.json`
- `sft_epoch1/{mmlu,truthfulqa_mc2,winogrande,hellaswag,gsm8k,arc_challenge}.json`

The full per-sample log files are still on the server under `/root/riskKD/output/.../` (not pulled — multi-GB per task).
