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

## ✅ DONE — kl_inv token weighting (α=1.0, 1 epoch)

Implemented and run. Token weighting **fixed the cross-size training instability**:

| Metric | no-weight (1ep) | **token-wt α=1.0 (1ep)** |
|---|---|---|
| `train_loss` (run-avg) | 5.40 | **0.98** |
| `radpo_kl/chosen` (mid-run) | ~1200–1600 | **~16–50** |
| `radpo_rewards/margins` | flips sign, mostly negative | **consistently positive (+0.5 to +1.9)** |
| `radpo_rewards/accuracies` | ~0.375 (below baseline) | **0.5–0.75 (above baseline)** |

**Benchmark eval** (lm-eval, vLLM TP=8):

| Benchmark | no-weight (1ep) | token-wt α=1.0 (1ep) | Δ |
|---|---|---|---|
| MMLU | 34.38% | 33.78% | −0.60 |
| TruthfulQA-MC2 | 49.06% | 48.14% | −0.92 |
| Winogrande | 60.85% | 61.80% | +0.95 |
| HellaSwag (acc_norm) | 67.74% | 67.82% | +0.08 |
| GSM8K (strict) | 3.94% | 6.22% | **+2.28** |
| ARC-C (acc_norm) | 38.74% | 41.13% | **+2.39** |
| **Average** | **42.45%** | **43.15%** | **+0.70** |

**Read:** modest net +0.70 avg, concentrated on the reasoning-ish tasks (GSM8K, ARC). Knowledge/recall tasks unchanged-to-slightly-down (within noise). The token weighting's clear win is *making riskKD trainable at all in the cross-size case* (KL 1000+→~30, loss 5.4→0.98) — that's a methods contribution even if the benchmark table is near-flat. Model saved at `riskKD_output_tokenwt/` (not pushed to HF).

Implementation: `CustomDPOConfig` has `radpo_token_weight_mode` (`"none"`|`"kl_inv"`) + `radpo_token_weight_alpha`. In `_radpo_get_batch_logps`, `kl_inv` mode builds `w_t = exp(-α · per_position_KL_t.detach())`. NOTE: α=1.0 is right because per-token KL is ~0.5–5 (NOT the summed-sequence KL ~1000 — α=0.001 was a no-op).

## ⏳ NEXT EXPERIMENT — 3-epoch token-weighted riskKD (launched but box died ~immediately; needs relaunch)

`riskKD.yaml` updated locally: `num_train_epochs: 3`, `save_strategy: "epoch"` (so we get checkpoint-902/1804/2706 — one per epoch, can eval the trajectory), `output_dir: .../riskKD_output_tokenwt_3ep`. Everything else unchanged (α=1.0 kl_inv, 8B `dpo_teacher_epoch1` ref, ultrafeedback, β=0.1, lr=5e-7 const, max_len=512, bsz=8). ~2706 steps × ~2.1 s/step ≈ ~95 min.

**To relaunch when box returns:**
```bash
scp -P 32305 recipes/.../riskKD.yaml root@<box>:/root/riskKD/recipes/llama3.2-1b-deita-dpomix/
ssh ... 'cd /root/riskKD && source .venv/bin/activate && export HF_TOKEN=$(cat /workspace/.hf_home/token) && \
  setsid bash -c "HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info DS_SKIP_CUDA_CHECK=1 \
    nohup python -m accelerate.commands.launch --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
    scripts/run_distill_dpo.py recipes/llama3.2-1b-deita-dpomix/riskKD.yaml > logs/riskKD_tokenwt_3ep.log 2>&1 < /dev/null" & disown -a'
```
Then set up the auto-eval watcher (MUST `cd /root/riskKD` first so the `logs/` redirect resolves) — wait for the training PID to disappear, then `bash eval_script/all.sh /home/.../riskKD_output_tokenwt_3ep`.

### --- old: kl_inv NEXT-EXPERIMENT section (kept for reference) ---

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

## Improvement ideas (prioritized)

Rating key — **P0** = do next / highest-leverage, **P1** = strong, do soon, **P2** = worthwhile, **P3** = nice-to-have / situational. "Methods" = changes the paper's contribution; "Exp" = a run/ablation, no method change.

| # | Idea | Type | Priority | Notes |
|---|---|---|---|---|
| 1 | **Derive the per-token weights from the augmented Pb-MDP** — the paper already has unspecified `w_t` in Eq. 10 (`w^w_i`, `w^l_j`) and Eq. 12. Argue those `w_t` ARE the importance-sampling correction the framework requires (since π_θ≠π_tch, any expectation under π_tch evaluated on π_θ's trajectory needs an IS weight; under a KL-trust-region assumption it reduces to `exp(-α·KL_t)`). Turns "we added a heuristic" → "the per-token weights in the ARR objective are the natural IS correction, which makes large-capacity-gap distillation tractable." The empirical KL 1000+→~30 / loss 5.4→0.98 result is the *evidence*. | Methods | **P0** | Biggest single win for the paper. No new code — reframes the existing `kl_inv` weighting as theory. |
| 2 | **Unify the token weighting with ARR's KL regularizer** — ARR has `−β·KL(π_θ‖π_tch)` (Eq. 5). The `exp(-α·KL_t)` weighting is *also* about KL. Frame as **per-token trust region**: only trust the teacher's value guidance where the student is close enough to act on it. One coherent mechanism instead of "global KL penalty + separate per-token hack." | Methods | **P0** | Pairs with #1. Strengthens the narrative; possibly lets you drop the global β in favor of the per-token treatment (ablate). |
| 5 | **Adaptive / annealed risk operator μ** — `radpo_confidence_level` is fixed at 0.99 (extreme tail-focus). Early in training (student far from teacher) aggressive risk-aversion is counterproductive. Anneal μ: start near risk-neutral, become risk-averse as training progresses (or make μ state-dependent). Fits the "nested" framing, gives a curriculum, clean ablation (static vs annealed μ). | Methods + Exp | **P0** | Needs a small code change: thread a μ-schedule into `_calculate_cvar_radpo` via a step-fraction. New YAML fields like `radpo_mu_anneal_start`, `radpo_mu_anneal_end`. |
| 6a | **3-epoch token-weighted run** — `riskKD.yaml` already updated (3 ep, `save_strategy: epoch` → per-epoch ckpts, `output_dir: riskKD_output_tokenwt_3ep`). Launched once, box died; needs relaunch. ~95 min. | Exp | **P1** | Cheapest concrete next run. Lets the preference signal accumulate; per-epoch ckpts show the trajectory. |
| 6b | **α sweep for kl_inv** — {0.5, 1.0, 2.0, 5.0}. α=1.0 worked; find the attenuation knee (too-small = no-op like α=0.001 was; too-large = throws away signal). | Exp | **P1** | ~30 min each; run after 6a. |
| 4t | **Better teacher → better value function** — `dpo_teacher_epoch1` was DPO'd from an *epoch-1 SFT whose held-out loss was rising* — mediocre teacher. Paper's own Limitations: "quality of the shaping signal depends on reliability of teacher value estimates." Re-DPO the 8B from a properly-trained SFT (the 3-epoch one or better) → bigger `Ṽ_πtch` quality → bigger transferred signal → benchmark deltas that actually show. | Exp | **P1** | ~30 min to re-DPO + the riskKD re-run. Most likely to move the benchmark table. |
| 3t | **3B-teacher → 1B-student ablation** — the 8B→1B explosion *is* the large-capacity-gap problem; token weighting treats the symptom. Use `meta-llama/Llama-3.2-3B-Instruct` (or DPO a 3B) as teacher; ARR should work *directly* (KL bounded). Story becomes: "ARR works directly at moderate gaps; the IS-weighted variant extends it to large gaps." | Exp | **P2** | Needs a 3B teacher (download Instruct, or DPO). Cleaner result than "8B→1B with a fix." |
| 6c | **Surgical weighting** — weight the *risk correction δ* by `exp(-α·KL)` but leave the *utility gap u* unweighted (the DPO signal is fine; only the risk-operator-on-garbage needs damping). More targeted than the current "multiply every per-token term uniformly." | Methods + Exp | **P2** | Small code change in `radpo_loss_fn` / `_radpo_get_batch_logps` — separate the weight applied to risk-ratio vs margin terms. Quick ablation. |
| 3p | **`pos_decay` token weighting** — `w_t = γ^(L-t)`, emphasize later (substantive) tokens. Alternative to `kl_inv`. | Exp | **P3** | Already mostly plumbed (the `radpo_token_weight_mode` enum) — would just need a `pos_decay` branch. |
| 3s | **Teacher-temperature softening** — divide teacher logits by T>1 in `_radpo_get_batch_logps` before the softmax. Smooths the 8B's peaks so the 1B can track. Single hyperparameter; orthogonal to token weighting (could stack). | Exp | **P3** | Not implemented. ~5-line change. |
| Bx | **Paper baselines** (none run yet): DPO on student (no teacher); TVKD (`qadapter_distil_weight=1`, `radpo_weight=0`); Ra-DPO-without-teacher (`ref = student_sft_init`). Needed for a complete experiments table regardless of method improvements. | Exp | **P1** | ~30 min each. |
| Ev | **Eval alignment quality (AlpacaEval 2 / MT-Bench win-rate)** — lm-eval benchmarks barely move under DPO by design; ARR targets instruction-following. Currently flying blind on the actual objective. | Exp | **P2** (user has opted to keep the lm-eval suite for now) | Would back the "ARR improves alignment" claim that the benchmark table can't. |

**Recommended order:** #1 + #2 + #5 (methods reframing + annealed μ code) → 6a (3-epoch run, already queued) → 4t (better teacher) → 6b (α sweep) → Bx (baselines). #1 and #2 are pure writing/reframing; #5 needs a small code change; the rest are runs.

## Repo state

- Branch `experiments/training-runs` (commits: `1689246` configs+eval_script+results, `e4b7780` loss-weighting proposal). `main` is at upstream `7e976f7`.
- Local `results/eval/` has SUMMARY.md + per-task JSONs for sft_epoch1, dpo_from_epoch1, riskKD.
- `proposals/loss-weighting.md` has the full menu of 6 weighting strategies.
- `eval_script/` has the lm-eval wrappers (mmlu, arc, gsm8k, hellaswag, wino, qa, all.sh — TP=8, gpu_memory_utilization=0.8).
- `alignment/configs.py` has the `attn_implementation` field added (was a missing-field regression in upstream that broke `run_sft.py`).
