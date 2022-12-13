#!/bin/bash
kubectl create secret generic $EXPERIMENT_NAME_PREFIX \
    --from-literal=WANDB_API_KEY=$WANDB_API_KEY --from-literal=HF_TOKEN=$HF_TOKEN
