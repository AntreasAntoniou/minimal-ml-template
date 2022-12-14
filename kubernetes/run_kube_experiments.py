import copy
import itertools
import os
from pathlib import Path
from typing import List
from rich import print


def get_scripts(exp_name: str, seeds: List[int]):

    script_list = []
    for seed in seeds:
        current_script_text = f"conda run -n main --live-stream /bin/bash /app/entrypoint.sh; conda run -n main --live-stream accelerate-launch --mixed_precision=bf16 /app/mlproject/run.py exp_name={exp_name} train_batch_size=300 eval_batch_size=300 seed={seed}"
        script_list.append(current_script_text)

    return script_list


if __name__ == "__main__":
    from bwatchcompute.kubernetes.job import Job

    script_list = get_scripts(
        exp_name=os.getenv("EXPERIMENT_NAME_PREFIX"), seeds=[42, 5, 10]
    )

    exp = Job(
        name=os.getenv("EXPERIMENT_NAME_PREFIX"),
        script_list=script_list,
        docker_image_path=os.getenv("DOCKER_IMAGE_PATH"),
        secret_variables={os.getenv("EXPERIMENT_NAME_PREFIX"): "WANDB_API_KEY"},
        environment_variables={
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "HF_USERNAME": os.getenv("HF_USERNAME"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "EXPERIMENTS_DIR": os.getenv("EXPERIMENTS_DIR"),
            "EXPERIMENT_DIR": os.getenv("EXPERIMENT_DIR"),
            "DATASET_DIR": os.getenv("DATASET_DIR"),
            "MODEL_DIR": os.getenv("MODEL_DIR"),
            "PROJECT_DIR": os.getenv("PROJECT_DIR"),
        },
        num_repeat_experiment=3,
    )

    exp.generate_spec_files()
    output = exp.run_jobs()
    print(output)
