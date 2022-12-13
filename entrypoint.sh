#!/bin/bash

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"

cd /app/
/opt/conda/envs/main/bin/pip install /app/
git pull

/bin/bash
