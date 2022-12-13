#!/bin/bash

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN >"~/.huggingface/token"

cd /workspace/minimal-ml-template/
/opt/conda/envs/main/bin/pip install -e /workspace/minimal-ml-template/
git pull

/bin/bash
