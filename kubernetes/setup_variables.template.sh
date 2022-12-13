#!/bin/bash
export WANDB_API_KEY=03f97dc1ae484a463ef95ba2e03fb12f7eeddd85
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=simple-kubernetes-tutorial

export HF_USERNAME=evolvingfungus
export HF_TOKEN=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

export EXPERIMENTS_DIR=/volume/experiments
export EXPERIMENT_DIR=/volume/experiments

export DATASET_DIR=/volume/datasets
export MODEL_DIR=/volume/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export USER_EMAIL="iam@antreas.io"
export USER_NAME="Antreas Antoniou"

export EXPERIMENT_NAME_PREFIX="debug-kube-0"
export DOCKER_IMAGE_PATH="ghcr.io/bayeswatch/minimal-ml-template:latest"

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"
