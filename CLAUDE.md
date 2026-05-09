# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research code for **TVKD (Teacher Value-Based Knowledge Distillation)** — the implementation accompanying *Preference Distillation via Value-Based Reinforcement Learning* (NeurIPS 2025). The `riskKD/` extension layers a **Risk-aware DPO (Ra-DPO) / CVaR** loss on top of TVKD. Built on top of HuggingFace `trl`, `transformers`, `peft`, `accelerate`/DeepSpeed, and `vllm`.

## Environment

Dependencies live in `requirments.txt` (sic, typo is the actual filename). Recommended pinned versions per the README:
- `torch==2.5.1`, `trl==0.12.0`, `peft==0.13.0`
- Plus `huggingface-hub`, `vllm`, `deepspeed`

Install: `pip install -r requirments.txt` (note the typo).

## Pipeline (3 stages)

The full TVKD/riskKD recipe is a sequence; running step N requires the artefacts produced by step N−1.

1. **Teacher SFT** → `scripts/run_sft.py` with `recipes/.../teacher_sft.yaml` (initial supervised fine-tuning of the teacher).
2. **Teacher DPO** → `scripts/run_distill_dpo.py` with `recipes/.../teacher_dpo.yaml` (DPO-train the teacher; required because TVKD assumes a DPO-trained teacher).
3. **Student SFT init** → `scripts/run_sft.py` with `recipes/.../student_sft_init.yaml`.
4. **(Optional) Precompute teacher logits** → `run/dckd*.sh` / `run/precompute.sh` invoke `utils/precompute_logits.py` then `utils/merge_logits_dckd_dataset.py` to build a dataset like `dpomix7k-dckd` containing `chosen_compressed_probs` / `rejected_compressed_probs`. Required for distillation losses that consume teacher token-level distributions (DCKD, vanila-KD, ADPA).
5. **TVKD / riskKD training** → `run/run_tvkd.sh` (TVKD) or `run/riskkd.sh` (riskKD) — both call `scripts/run_distill_dpo.py` with the matching YAML.

The "with_ref" and "with_teacher" variants (`scripts/run_distil_dpo_with_ref.py`, `scripts/run_distill_dpo_with_teacher.py`) are alternative trainers — they share the same `CustomDPOConfig` / `DistillTrainer` skeleton as `run_distill_dpo.py` but differ in how the reference / teacher model is loaded and used. `scripts/run_distill_dpo.py` is the primary entry point referenced by all the run scripts shipped in `run/`.

## Common commands

All training launches go through `accelerate.commands.launch`; the `run/*.sh` scripts are the canonical invocations. Examples:

Teacher SFT:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_sft.yaml
```

TVKD / riskKD (the launchers in `run/`):
```bash
./run/run_tvkd.sh    # TVKD with recipes/.../TVKD.yaml
./run/riskkd.sh      # riskKD (TVKD + Ra-DPO) with recipes/.../riskKD.yaml
```

Override any YAML field on the command line (parser logic lives in `alignment.configs.H4ArgumentParser.parse_yaml_and_args`):
```bash
python -m accelerate.commands.launch ... scripts/run_distill_dpo.py \
  recipes/.../riskKD.yaml --learning_rate=1e-6 --beta=0.05
```

Precompute teacher logits (run before any DCKD / vanila-KD / ADPA training):
```bash
bash run/dckd.sh                # dpo-mix-7k + LLaMA teacher
bash run/dckd_helpsteer2.sh     # HelpSteer2_DPO
bash run/dckd_mistral.sh        # Mistral teacher
```

Iterative offline DPO (separate experiment loop, not part of TVKD): `bash run/redo.sh` — runs `scripts/offline_generation.py` (vLLM-style sampling + reward model scoring with `sfairXC/FsfairX-LLaMA3-RM-v0.1`) and `scripts/dpo_training.py` over N cycles, where each cycle's checkpoint becomes the next cycle's student.

There is no test suite, lint config, or CI; `scripts/test_files.py` and `scripts/test_files.ipynb` are ad-hoc inspection utilities, not unit tests.

## Architecture: how the trainer composes losses

`scripts/run_distill_dpo.py` (~2900 lines) is the heart of the codebase. The training objective is **a weighted sum** assembled at runtime in `DistillTrainer.get_batch_loss_metrics` (~line 1636). Each component is gated by a YAML weight; setting a weight to 0 disables it. To change which "method" you're running, you don't switch trainer files — you change the weights:

| Loss term | Weight field | Implementing call |
|---|---|---|
| Chosen / rejected NLL (SFT) | `sft_on_chosen`, `sft_on_rejected` | inline cross-entropy on `model_output["nll_loss"]` |
| Standard DPO | `dpo_weight` | `self.dpo_loss(...)` (with `loss_type` selector — `sigmoid`, `ipo`, `nca_pair`, etc.) |
| **Ra-DPO / CVaR (riskKD)** | `radpo_weight` (+ `radpo_alpha`, `radpo_confidence_level`, `if_radpo2`, `is_split_risk_ratio`, `is_cal_risk_distribution_logps`) | `radpo_concatenated_forward` → `_calculate_cvar_radpo` / `_cal_risk_distribution_logps_radpo` → `radpo_loss_fn` |
| Token-level distillation (DCKD / vanila-KD) | `distillation_weight`, `chosen_distil_weight`, `rejected_distil_weight` | uses precomputed `teacher_chosen_probs` / `teacher_rejected_probs` via `utils.compress_logits.load_input_and_target_probs_*` |
| ADPA margin distillation | `adpa_weight` (+ `adpa_loss_type` ∈ {`reverse_ce`, `soft_kl`}, `adpa_temperature`) | needs `rejected_margin_logp_every` column |
| **TVKD Q-adapter** | `qadapter_distil_weight` (+ `qadapter_alpha`, `qadapter_alpha_tilde`, `qadapter_alpha_1`, `qadapter_loss_type`, `qadapter_softvalue_type` ∈ {`sum`,`entropy`,...}, `qadapter_with_ref`, `qadapter_gamma`, `qadapter_reward_normalization`) | uses ref/teacher logps with the `shifted_v_fn` soft-value computation inside `get_batch_loss_metrics` |
| Sequence KD | `sequence_kd_weight` | needs `rejected_margin_logp_every` |
| KL student-vs-ref regularizer | `kl_student_weight` (+ `kl_student_target` ∈ {`chosen`,`rejected`,`both`}) | direct KL on `ref_*_logps` |
| KL penalty | `kl_penalty_weight` | added near end of `get_batch_loss_metrics` |
| Other (`tpkd_*`, `dpkd_distil_weight`, `new_tpkd_weight`, `simpo_gamma`, `dpo_length_normalization`, `discopop_tau`, `sync_ref_model`, `rpo_alpha`, `use_weighting`) | various | exposed in `CustomDPOConfig` for ablations |

**Practical implication:** the recipe YAMLs are the experiment definition. `TVKD.yaml` zeroes out DPO and uses `qadapter_distil_weight=1`; `riskKD.yaml` adds `radpo_weight=1`; `vanilakd.yaml` uses `distillation_weight=1` with `chosen_distil_weight=0.1`; `wpo.yaml` enables `use_weighting=true`; `teacher_dpo.yaml` is pure DPO (`dpo_weight=1`, all distillation weights 0). When asked to "implement method X," first check whether X is just a different weight combination of these existing terms — it usually is.

The trainer also branches on which extra dataset columns are present. `main()` builds `columns_to_keep` conditionally based on the active weights:
- `chosen_distil_weight > 0 and distillation_weight > 0` → loads `teacher_chosen_probs`
- `rejected_distil_weight > 0 and distillation_weight > 0` → loads `teacher_rejected_probs`
- `adpa_weight > 0` → loads `rejected_margin_logp_every`

If you enable a weight whose required column isn't in your dataset, `mix_datasets` will silently drop that column and the loss term will fail at runtime. Make sure the precompute pipeline (`utils/precompute_logits*.py` + `utils/merge_logits_*_dataset.py`) was run with the right teacher and merged into the dataset path referenced by `dataset_mixer` in the YAML.

`mix_datasets` in `run_distill_dpo.py` (line 252) also auto-renames `chosen_compressed_probs` → `teacher_chosen_probs` and `rejected_compressed_probs` → `teacher_rejected_probs` so that precomputed datasets line up with what the trainer expects.

## Configuration system

- **`alignment/configs.py`** defines `H4ArgumentParser` (a `HfArgumentParser` subclass), `ModelArguments`, `DataArguments`, `SFTConfig`, `DPOConfig`. The parser accepts a single YAML path as `sys.argv[1]` and treats remaining args as overrides. This is why every entry-point script just does `parser.parse()`.
- Trainer-specific knobs (the table above) live in `CustomDPOConfig` inside `scripts/run_distill_dpo.py`, NOT in `alignment/configs.DPOConfig`. If you add a new YAML field, add it to that local dataclass.
- DeepSpeed configs in `recipes/accelerate_config/` differ in `num_processes` (`deepspeed_zero3.yaml` = 2, `_2.yaml` = 2, `_3.yaml` ≈ 3, `_8.yaml` ≈ 8). Pick the one matching `CUDA_VISIBLE_DEVICES` count or override.
- The recipe YAMLs in this repo contain absolute paths from a previous user (`/home/minchan.kwon/...`, `/mnt/datpv/...`); update `model_name_or_path`, `ref_model_name_or_path`, `dataset_mixer` keys, and `output_dir` before running.

## Data flow

- `alignment.data.get_datasets` → `mix_datasets`: tries `load_dataset(name)` (HF Hub), falls back to `load_from_disk(os.path.join(name, split))` for local paths. `dataset_mixer` keys can therefore be either Hub names or local directories.
- `alignment.data.apply_chat_template` (used by SFT path) and the in-file `apply_chat_template` in `run_distill_dpo.py` (used by DPO path) are NOT identical — the DPO version requires OpenAI-format `chosen` / `rejected` and produces `text_prompt` / `text_chosen` / `text_rejected`, which are then renamed to `prompt` / `chosen` / `rejected` for TRL.
- The chat template applied to `meta-llama/Llama-3.2-1B` is hard-coded in the recipe YAMLs as a `chat_template:` string; no template = the tokenizer's default.

## Things to know before editing

- Top-of-file `sys.path.append("/home/minchan.kwon/ADPA")` / `sys.path.append("~/ADPA")` lines in `scripts/run_*.py` are remnants from the original author's environment. They do nothing useful here (the path doesn't exist on a fresh checkout) but also don't break — `alignment` is importable because the working directory is the repo root.
- Many comments in the YAMLs and Python files are in Korean (e.g. `#alpha 수정 버전`, `❗️ NaN 발생!`). They explain author intent; preserve them when editing.
- `wandb/` is a local run cache from prior training; ignore it. Logging to W&B is opt-in via `report_to: wandb` + `run_name:` in the YAML (see `riskKD.yaml`).
- The three "distill" trainer scripts (`run_distill_dpo.py`, `run_distil_dpo_with_ref.py`, `run_distill_dpo_with_teacher.py`) share large amounts of duplicated code. A change to the loss in one usually needs to be mirrored in the others if those entry points are still in use — but the run scripts in `run/` only call `run_distill_dpo.py`, so prefer that one as the source of truth.
- `analyze_margin.py` (~2200 lines) is offline analysis tooling, not part of the training loop.
