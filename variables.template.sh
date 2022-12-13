#!/bin/bash
export WANDB_API_KEY=api-key
export WANDB_ENTITY=wandb-username
export WANDB_PROJECT=simple-tutorial

export HF_USERNAME=hf-username
export HF_TOKEN=my-hf-token

export EXPERIMENTS_DIR=/volume/experiments
export EXPERIMENT_DIR=/volume/experiments

export DATASET_DIR=/volume/datasets
export MODEL_DIR=/volume/models

export PROJECT_DIR=/app/

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"
