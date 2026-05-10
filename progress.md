# Progress — riskKD / Risk-Aware Preference Distillation experiments

Last updated: 2026-05-10

## Context

- **Repo**: `riskKD` = the codebase for TVKD (NeurIPS 2025, arxiv 2509.16965) + an Ra-DPO/CVaR extension.
- **Our proposal** (`_EMNLP_2026__Risk_Aware_Preference_Distillation.pdf`): **NRPD / ARR** — Nested Risk-Aware Preference Distillation via Teacher Value Shaping. Builds on TVKD by adding a nested risk operator Φ^μ (CVaR) to the teacher's value function. Final objective: `L_ARR = -E[log σ(u − δ)]` where `u` is the token-weighted DPO-style log-ratio (student vs teacher) and `δ` is the sequential risk correction.
- **Compute**: vast.ai box, 8× H200 (144 GB each). `/root/riskKD` is the working copy; `.venv` has torch 2.5.1+cu124, trl 0.12.0, peft 0.13.0, transformers 4.49.0, accelerate 1.4.0, deepspeed 0.19.0, vllm 0.6.6.post1, lm-eval 0.4.11.
- **HF account**: pushes go to `vukien2301/*` (private). The `pvdhihihi`-credentialed token (`hf_kfiU...`) has gated access to `meta-llama/Llama-3.x` models; the `vukien2301` token is the default in `/workspace/.hf_home/token` (used for pushes). `HF_HOME=/workspace/.hf_home` on the box — write tokens there, not `~/.cache/huggingface/token`.

## What's been trained

| Model | What it is | On HF? | On-box path |
|---|---|---|---|
| `llama-3.1-8b-deita-sft-teacher` | 8B Llama-3.1 + 6-epoch SFT on deita-10k | ✅ | (deleted from box) |
| `llama-3.1-8b-deita-sft-teacher-epoch1` | 8B + 1-epoch SFT (the DPO base) | ✅ | (in `ref_teacher_3epochs/checkpoint-191`) |
| `llama-3.1-8b-dpomix-dpo-teacher` | 8B, cosine-LR DPO on dpo-mix-7k from 6-ep SFT | ✅ | (deleted) |
| `llama-3.1-8b-ultrafeedback-dpo-teacher` | 8B, constant-LR DPO on ultrafeedback from 6-ep SFT | ✅ | (deleted) |
| **`llama-3.1-8b-ultrafeedback-dpo-from-epoch1`** | 8B, constant-LR DPO on ultrafeedback **from epoch-1 SFT** — used as the riskKD teacher | ✅ | `dpo_teacher_epoch1/` |
| **`llama-3.2-1b-deita-sft-student`** | **1B Llama-3.2 + 3-epoch SFT on deita-10k** — the riskKD student SFT init | ✅ | `student_sft_init/` |
| riskKD output (Ra-DPO, no token weighting) | 1B student trained with Ra-DPO loss vs 8B DPO teacher, ultrafeedback, β=0.1, lr=5e-7 const, max_len=512, 1 epoch (902 steps) | ⬜ not pushed | `riskKD_output/` |

## What's been evaluated (lm-eval-harness, vLLM TP=8, gpu_memory_utilization=0.8, max_model_len=4096)

Results pulled to `results/eval/` (`SUMMARY.md` + per-task JSONs).

| Benchmark | Few-shot | Metric | SFT-ep1 (8B) | DPO-from-ep1 (8B) | riskKD-student (1B) |
|---|---|---|---|---|---|
| MMLU | 5 | acc | 63.05% | 63.39% | 34.38% |
| TruthfulQA-MC2 | 0 | acc | 50.59% | 57.37% | 49.06% |
| Winogrande | 5 | acc | 78.45% | 78.22% | 60.85% |
| HellaSwag | 10 | acc_norm | 81.85% | 83.05% | 67.74% |
| GSM8K | 5 | exact_match (strict) | 47.61% | 52.16% | 3.94% |
| ARC-Challenge | 25 | acc_norm | 51.19% | 52.30% | 38.74% |
| **Average** | | | **62.12%** | **64.42%** | **42.45%** |

**Read:** 8B SFT→DPO gave +2.3 avg (TruthfulQA +6.8, GSM8K +4.6). The 1B riskKD student is capacity-limited; the only signal Ra-DPO transferred was TruthfulQA +11 over base 1B (~38%→49%). Knowledge benchmarks unchanged (expected — DPO doesn't add knowledge). ARC-C low for all (chat-format vs raw-loglikelihood mismatch — would close with `--apply_chat_template`).

## Key finding from the riskKD run

Ra-DPO / ARR with a 1B student and 8B teacher is **numerically unstable**: `radpo_kl/chosen` and `radpo_kl/rejected` stayed ~800–1600 throughout (should be single digits), `radpo_rewards/margins` flipped sign, accuracies hovered at the 0.5 random baseline. Cause: the policy (1B) cannot match the reference (8B) distribution → the log-ratio `log(π_θ/π_tch)` is huge and noisy → the CVaR risk term swamps the preference signal. Loss only converged late and noisily (train_loss=5.40). Decision: **keep the 1B/8B setup** (it's a real, reportable finding) and try to **stabilize via token-level weighting** rather than changing the model pair.

## ⏳ NEXT EXPERIMENT — kl_inv token weighting (implemented, not yet launched; box was down)

**Idea** (from `proposals/loss-weighting.md` #4a): downweight tokens where policy & teacher disagree wildly — the cross-size noise tokens — inside the Ra-DPO loss.

```
w_t = exp(-α · per_position_KL_t)      # α = 0.001 → weight ≈ 0.37 at KL=1000, ≈0 at the wildest tokens
```

**Implementation** (done, in `scripts/run_distill_dpo.py`):
- `CustomDPOConfig`: new fields `radpo_token_weight_mode` (`"none"` | `"kl_inv"`) and `radpo_token_weight_alpha` (float).
- `_radpo_get_batch_logps`: accepts `token_weight_mode`, `token_weight_alpha`; when mode is `kl_inv`, builds `weights = exp(-α · per_position_kl.detach())` instead of all-ones. The existing `weights` plumbing already multiplies every per-token quantity (`logps_margin`, `per_position_kl`, `per_position_risk_ratio`, `per_token_logps`) — so high-KL tokens contribute less to all of them uniformly.
- `radpo_concatenated_forward`: threads the two config values through.
- `riskKD.yaml`: `radpo_token_weight_mode: kl_inv`, `radpo_token_weight_alpha: 0.001`, `output_dir: .../riskKD_output_tokenwt`. Everything else unchanged (1B student `student_sft_init`, 8B ref `dpo_teacher_epoch1`, ultrafeedback, β=0.1, lr=5e-7 const, max_len=512, bsz=8, 1 epoch).

**To launch when the box is back:**
```bash
scp -P 32305 scripts/run_distill_dpo.py recipes/.../riskKD.yaml root@<box>:/root/riskKD/...
ssh ... 'cd /root/riskKD && source .venv/bin/activate && \
  export HF_TOKEN=$(cat /workspace/.hf_home/token) && \
  setsid bash -c "HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info DS_SKIP_CUDA_CHECK=1 \
    nohup python -m accelerate.commands.launch --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
    scripts/run_distill_dpo.py recipes/llama3.2-1b-deita-dpomix/riskKD.yaml > logs/riskKD_tokenwt.log 2>&1 < /dev/null" & disown -a'
```
Then eval with `bash eval_script/all.sh /home/.../riskKD_output_tokenwt` and pull to `results/eval/riskKD_tokenwt/`.

**Success criteria:** `radpo_kl/*` should drop well below the ~1000 we saw; `radpo_rewards/margins` should stay positive; `radpo_rewards/accuracies` should rise above 0.5; benchmark scores should be ≥ the no-weighting riskKD run (especially TruthfulQA).

## Other queued / candidate experiments

1. **α sweep for kl_inv** — once 0.001 is validated, try {0.0005, 0.002, 0.005} to see how much attenuation is optimal.
2. **`pos_decay` token weighting** — `w_t = γ^(L-t)`, emphasize later (substantive) tokens. Alternative to kl_inv.
3. **Teacher-temperature softening** — divide teacher logits by T>1 in `_radpo_get_batch_logps` before the softmax. Smooths the 8B's peaks so the 1B can track. Single hyperparameter; not yet implemented.
4. **Baselines for the paper's experiments table** (none run yet):
   - DPO baseline: 1B `student_sft_init` → standard DPO on ultrafeedback, no teacher.
   - TVKD baseline: `qadapter_distil_weight=1`, `radpo_weight=0`, 8B teacher ref — the expectation-based teacher value shaping (the original TVKD method).
   - Ra-DPO-without-teacher: `radpo_weight=1` but `ref_model = student_sft_init` (the student's own SFT) instead of the 8B teacher.
5. **Eval the alignment objective, not just knowledge** — AlpacaEval 2 / MT-Bench win-rate. lm-eval benchmarks (MMLU etc.) barely move under DPO by design; the thing ARR actually targets is instruction-following quality. (Lower priority per current decision to keep the existing benchmark suite, but worth noting.)

## Repo state

- Branch `experiments/training-runs` (commits: `1689246` configs+eval_script+results, `e4b7780` loss-weighting proposal). `main` is at upstream `7e976f7`.
- Local `results/eval/` has SUMMARY.md + per-task JSONs for sft_epoch1, dpo_from_epoch1, riskKD.
- `proposals/loss-weighting.md` has the full menu of 6 weighting strategies.
- `eval_script/` has the lm-eval wrappers (mmlu, arc, gsm8k, hellaswag, wino, qa, all.sh — TP=8, gpu_memory_utilization=0.8).
- `alignment/configs.py` has the `attn_implementation` field added (was a missing-field regression in upstream that broke `run_sft.py`).
