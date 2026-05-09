---
library_name: transformers
license: llama3.1
base_model: meta-llama/Llama-3.1-8B
tags:
- alignment-handbook
- generated_from_trainer
datasets:
- HuggingFaceH4/deita-10k-v0-sft
model-index:
- name: ref_teacher
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ref_teacher

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) on the HuggingFaceH4/deita-10k-v0-sft dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1701

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 128
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 6

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.9397        | 1.0   | 286  | 0.9903          |
| 0.7682        | 2.0   | 572  | 0.9505          |
| 0.6005        | 3.0   | 858  | 0.9660          |
| 0.3653        | 4.0   | 1144 | 1.0276          |
| 0.1982        | 5.0   | 1430 | 1.1059          |
| 0.1315        | 6.0   | 1716 | 1.1701          |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.5.1+cu124
- Datasets 4.8.5
- Tokenizers 0.21.4
