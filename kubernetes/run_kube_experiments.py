import copy
import datetime
import itertools
import os
from pathlib import Path
from typing import List

from rich import print


def get_scripts(exp_name: str, batch_sizes: List[int]):
    script_list = []
    for batch_size in batch_sizes:
        current_script_text = f"/opt/conda/envs/main/bin/accelerate-launch --mixed_precision=bf16 /app/mlproject/run.py exp_name={exp_name} train_batch_size={batch_size} eval_batch_size={batch_size} code_dir=/app/"
        script_list.append(current_script_text)

    return script_list


if __name__ == "__main__":
    from bwatchcompute.kubernetes.job import Job

    script_list = get_scripts(
        exp_name=os.getenv("EXPERIMENT_NAME_PREFIX"),
        batch_sizes=[75, 150, 300],
    )
    # write a one liner that picks up date and time and converts them into a number
    datetime_seed = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    exp = Job(
        name=f"{datetime_seed}-{os.getenv('EXPERIMENT_NAME_PREFIX')}",
        script_list=script_list,
        docker_image_path=os.getenv("DOCKER_IMAGE_PATH"),
        secret_variables={
            os.getenv("EXPERIMENT_NAME_PREFIX"): "WANDB_API_KEY"
        },
        environment_variables={
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "HF_USERNAME": os.getenv("HF_USERNAME"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
            "CODE_DIR": os.getenv("CODE_DIR"),
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
