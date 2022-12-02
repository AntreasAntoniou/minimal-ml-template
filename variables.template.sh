#!/bin/bash
export WANDB_API_KEY="my-key"
export WANDB_ENTITY="username"
export WANDB_PROJECT="mini-ml-template"
export HF_TOKEN="hf-key"
export PROJECT_DIR="my-awesome-ml-project"

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"

git config --global credential.helper store
git config --global --add safe.directory $PROJECT_DIR
