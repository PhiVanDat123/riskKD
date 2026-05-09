---
library_name: transformers
base_model: vukien2301/llama-3.1-8b-deita-sft-teacher
tags:
- alignment-handbook
- generated_from_trainer
datasets:
- pvdhihihi/ultra-feedback
model-index:
- name: dpo_teacher_constant
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# dpo_teacher_constant

This model is a fine-tuned version of [/home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/ref_teacher](https://huggingface.co//home/minchan.kwon/ADPA/model/llama3.2-1b-deita-dpomix/ref_teacher) on the pvdhihihi/ultra-feedback dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7e-07
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 256
- total_eval_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results



### Framework versions

- Transformers 4.49.0
- Pytorch 2.5.1+cu124
- Datasets 4.8.5
- Tokenizers 0.21.4
