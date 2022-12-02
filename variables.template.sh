#!/bin/bash
export WANDB_API_KEY=""
export WANDB_ENTITY="machinelearningbrewery"
export WANDB_PROJECT="mini-ml-template"
export HF_TOKEN=""
export PROJECT_DIR=""

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"

git config --global credential.helper store
git config --global --add safe.directory $PROJECT_DIR
