# ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting
This repo contains the code for our paper [ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](https://arxiv.org/abs/2406.19976). In this work, we introduce ScaleBiO, the first scalable instantiation of first-order bilevel optimization algorithm, focusing on large-scale LLM data reweighting.

## Latest News
* [2024-07-30] Preview version - we release the demo data reweighting code and data for [gpt2](https://huggingface.co/openai-community/gpt2) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B) on alpaca and alpaca-gpt4 dataset.

## Quick Start

### Setup
```
pip install -r requirements.txt
```

### Reweighting
export WANDB_API_KEY=<your_wandb_api_key> and turn on --use_wandb in the script

`./run_gpt2.sh`

`./run_Yi34B.sh`

### Recommended Hardware Configuration
8x A40/A100 GPUs and 2TB Memory

## Citation
If you find this repository useful, please consider giving ⭐ and citing our [paper](https://arxiv.org/abs/2406.19976):
```
@article{pan2024scalebio,
  title={ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting},
  author={Pan, Rui and Zhang, Jipeng and Pan, Xingyuan and Pi, Renjie and Wang, Xiaoyu and Zhang, Tong},
  journal={arXiv preprint arXiv:2406.19976},
  year={2024}
}
```
