# minimal-stateless-ml-template

This repo implements a **minimal** machine learning template, that is fully featured for most of the things a machine learning project might need. The most important parts that set this repo apart from the rest are:

1. It is **stateless** any given experiment ran using this template automatically and periodically stores the model weights and configuration to [HuggingFace Hub](https://huggingface.co/docs/hub/models-the-hub) and [wandb](https://wandb.ai/site) respectively. As a result, if your machine dies or job exits, and you resume on another machine, the code will automatically locate and download the previous history and continue from where it left off. This makes this repo very useful when using spot instances, or using schedulers like slurm and kubernetes. 
2. It provides support for all the latest and greatest GPU and TPU optimization and scaling algorithms through [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index).
3. It provides mature configuration support via [Hydra-Zen](https://github.com/mit-ll-responsible-ai/hydra-zen) and automates configuration generation via [decorators](https://github.com/BayesWatch/minimal-ml-template/blob/af387e59472ea67552b4bb8972b39fe95952dd8a/mlproject/decorators.py#L10) implemented in this repo.
4. It has a minimal **callback** based boilerplate that allows a user to easily inject any functionality at predefined places in the system without spagettifying the code.
5. It uses [HuggingFace Models](https://huggingface.co/models) and [Datasets](https://huggingface.co/docs/datasets/index) to streamline building/loading of models, and datasets, but is also not forcing you to use those, allowing for very easy injection of any models and datasets you care about, assuming you use models implemented under PyTorch's `nn.Module` and `Dataset` classes.
6. It provides plug and play functionality that allows easy hyperparameter search on Kubernetes clusters using [BWatchCompute](https://github.com/BayesWatch/bwatchcompute) and some readily available scripts and yaml templates.

## The Software Stack

## Getting Started

### Installation

## Running experiments locally

## Modify the experiment configuration from the command line

## Tracking experiments with wandb

## Setting up the usage of huggingface model and dataset hubs so you can store your model weights and datasets

## Making any class and/or function configurable via Hydra-Zen

## Adding a new model

## Adding a new dataset

## Adding a new callback

## Running a kubernetes hyperparameter search
