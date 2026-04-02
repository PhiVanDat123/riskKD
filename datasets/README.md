---
language:
- en
license: mit
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: dataset
    dtype: string
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: chosen_rating
    dtype: float64
  - name: rejected_rating
    dtype: float64
  splits:
  - name: train
    num_bytes: 41362946
    num_examples: 6750
  - name: test
    num_bytes: 4586808
    num_examples: 750
  download_size: 24232011
  dataset_size: 45949754
tags:
- distilabel
- synthetic
- dpo
- argilla
---

# Argilla DPO Mix 7K Dataset

> A small cocktail combining DPO datasets built by Argilla with [distilabel](https://github.com/argilla-io/distilabel). The goal of this dataset is having a small, high-quality DPO dataset by filtering only highly rated chosen responses. 

<div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/Csd2-zPji7iwIxyz6UFe1.webp">
</div>


<p align="center">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>


## Datasets mixed

As already mentioned, this dataset mixes the following datasets:

* [`argilla/distilabel-capybara-dpo-7k-binarized`](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized): random sample of highly scored chosen responses (>=4).
* [`argilla/distilabel-intel-orca-dpo-pairs`](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs): random sample of highly scored chosen responses (>=8).
* [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned): random sample of highly scored chosen responses (>=4).

The samples have been randomly selected from the original datasets with a proportion of 0.33 each, as can be seen via the `dataset` column of the dataset.

## Next steps

* Adding more samples
* Use data selection techniques to improve the diversity, usefulness, and complexity of the dataset.