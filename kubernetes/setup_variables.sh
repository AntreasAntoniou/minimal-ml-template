#!/bin/bash
export WANDB_API_KEY=blablabla
export WANDB_ENTITY=team-name
export WANDB_PROJECT=minimal-ml-template-project

export HF_USERNAME=username
export HF_TOKEN=blablabla

export CODE_DIR=/app/
export EXPERIMENT_NAME_PREFIX="batch-size-search-v-2-0"
export EXPERIMENTS_DIR=/volume/experiments
export EXPERIMENT_DIR=/volume/experiments
export DATASET_DIR=/volume/datasets
export MODEL_DIR=/volume/models

export CLUSTER_NAME=spot-gpu-cluster-1
export CLUSTER_ZONE=us-central1-a
export CLUSTER_PROJECT=tali-multi-modal

export DOCKER_IMAGE_PATH="ghcr.io/bayeswatch/minimal-ml-template:latest"
