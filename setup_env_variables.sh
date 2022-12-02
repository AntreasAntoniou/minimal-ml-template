export WANDB_API_KEY=9a295b91cf65c412f681e492d6b92fe4be4b000b
export WANDB_ENTITY=machinelearningbrewery
export WANDB_PROJECT=mini-ml-template
export HF_TOKEN=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
export PROJECT_DIR=/workspaces/minimal-ml-template

mkdir -p "~/.huggingface"
touch "~/.huggingface/token"

echo $HF_TOKEN > "~/.huggingface/token"

git config --global credential.helper store
git config --global --add safe.directory $PROJECT_DIR