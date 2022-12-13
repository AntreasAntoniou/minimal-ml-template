#!/bin/bash
export WANDB_API_KEY=my-api-key
export WANDB_ENTITY=my-wandb-username
export WANDB_PROJECT=simple-tutorial

export HF_USERNAME=my-hf-username
export HF_TOKEN=my-hf-api-token

export EXPERIMENTS_DIR=/volume/experiments
export EXPERIMENT_DIR=/volume/experiments

export DATASET_DIR=/volume/datasets
export MODEL_DIR=/volume/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export EXPERIMENT_NAME_PREFIX="debug-kube-0"
export DOCKER_IMAGE_PATH="ghcr.io/bayeswatch/minimal-ml-template:latest"

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"
