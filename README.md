# minimal-stateless-ml-template

![image](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![image](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<img src="https://github.com/mit-ll-responsible-ai/hydra-zen/blob/main/brand/Hydra-Zen_logo_full.svg" alt= “” height="32">
<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt= “” height="32">

This repo implements a **minimal** machine learning template, that is fully featured for most of the things a machine learning project might need. The most important parts that set this repo apart from the rest are:

1. It is **stateless**. Any given experiment ran using this template, will, automatically and periodically stores the model weights and configuration to [HuggingFace Hub](https://huggingface.co/docs/hub/models-the-hub) and [wandb](https://wandb.ai/site) respectively. As a result, if your machine dies or job exits, and you resume on another machine, the code will automatically locate and download the previous history and continue from where it left off. This makes this repo very useful when using spot instances, or using schedulers like slurm and kubernetes. 
2. It provides support for all the latest and greatest GPU and TPU optimization and scaling algorithms through [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index).
3. It provides mature configuration support via [Hydra-Zen](https://github.com/mit-ll-responsible-ai/hydra-zen) and automates configuration generation via [decorators](https://github.com/BayesWatch/minimal-ml-template/blob/af387e59472ea67552b4bb8972b39fe95952dd8a/mlproject/decorators.py#L10) implemented in this repo.
4. It has a minimal **callback** based boilerplate that allows a user to easily inject any functionality at predefined places in the system without spagettifying the code.
5. It uses [HuggingFace Models](https://huggingface.co/models) and [Datasets](https://huggingface.co/docs/datasets/index) to streamline building/loading of models, and datasets, but is also not forcing you to use those, allowing for very easy injection of any models and datasets you care about, assuming you use models implemented under PyTorch's `nn.Module` and `Dataset` classes.
6. It provides plug and play functionality that allows easy hyperparameter search on Kubernetes clusters using [BWatchCompute](https://github.com/BayesWatch/bwatchcompute) and some readily available scripts and yaml templates.

## The Software Stack

This machine learning project template is built using the following software stack:
1. Deep Learning Framework: [PyTorch](https://pytorch.org/get-started/locally/)
2. Dataset storage and retrieval: [Huggingface Datasets](https://huggingface.co/docs/datasets/index)
3. Model storage and retrieval [Huggingface Hub](https://huggingface.co/docs/hub/models-the-hub), and [HuggingFace Models](https://huggingface.co/models)
4. GPU/TPU/CPU Optimization and Scaling up options library: [Huggingface Accelerate](https://huggingface.co/docs/accelerate/index)
5. Experiment configuration + command line argument parsing: [Hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen)
6. Experiment tracking: [Weights and Biases](https://docs.wandb.ai)
7. Simple python based ML experiment running with Kubernetes using [BWatchCompute](https://github.com/BayesWatch/bwatchcompute)

## Getting Started

### Installation

There are two supported options available for installation.

1. Using conda/mamba
2. Using docker

## Install via conda
To install via conda:
1. Clone the template
```bash
git clone https://github.com/AntreasAntoniou/minimal-ml-template/
``` 
2. Run:
```bash
bash -c "source install-via-conda.sh"
```

If you do not have conda installed it will be installed for you. If you do, it'll simply install the necessary dependencies in an environment named minimal-ml-template

## Install via docker
You can use a docker image to get a full installation of all relevant dependencies and a copy of the template to get started.

**Note: We recommend using VSCode with the docker extension so that you can attach your IDE to the python environment within the docker container and develop directly, as explained in https://code.visualstudio.com/docs/devcontainers/containers**.

To install via docker:

1. Install docker on your system, and start the docker daemon.
2. `docker pull docker pull ghcr.io/antreasantoniou/minimal-ml-template:latest`
3. `docker run --gpus all --shm-size=<RAM-AVAILABLE> -it ghcr.io/antreasantoniou/minimal-ml-template:latest`. Replacing <RAM-AVAILABLE> with the amount of memory you want the docker container to utilize. 
4. (Optional) If you wish to be able to modify the codebase and keep a copy of it available in the local filesystem, then first clone the repository to a local directory of your choosing and then simply use `docker run --gpus all -v path/to/local/repo/clone:/repo/ --shm-size=<RAM-AVAILABLE> -it ghcr.io/antreasantoniou/minimal-ml-template:latest` and `cd /repo/` to enter the linked directory, and then simply run `pip install -e .` to install the repo in a development mode so any changes you make will be reflected in the mlproject package.

## Setting up the relevant environment variables
Before running any experiment, you must set the environment variables necessary for huggingface and wandb to work properly, as well as the environment variables for the necessary directories in which to store datasets, models and experiment tracking. 

A template for the necessary variables is available in [run.env](run.env). To modify:
1. Open with your favourite file editor
2. Fill in the wandb key as explained in https://wandb.ai/authorize
3. Fill the hugging face username and access token as explained in https://huggingface.co/settings/tokens
4. Fill in the paths in which to store datasets/models etc
5. Run `source run.env` to load the environment variables

## Running experiments locally
**Note**: Before running any experiments, you must set the environment variables as explained in the previous section.

Running experiments on a local machine can be done by issuing the following command to the command line:
```bash
accerate launch mlproject/run.py exp_name=my-awesome-experiment-0
```
The above can also be achieved by replacing `accelerate launch` with `python` but using accelerate launch means all the awesome compute optimizations that the Accelerate framework provides can be engaged. 

To get a full list of the arguments that the minimal-ml-template framework can receive use:
```bash
accelerate launch mlproject/run.py --help
```

<details>
    <summary>View default response</summary>

```bash
run is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

callbacks: default
dataloader: default
dataset: food101
learner: default
model: vit_base_patch16_224
optimizer: adamw
scheduler: cosine-annealing
wandb_args: default


== Config ==
Override anything in the config (foo.bar=value)

exp_name: ???
model:
  _target_: mlproject.models.build_model
  model_name: google/vit-base-patch16-224-in21k
  pretrained: true
  num_classes: 101
dataset:
  _target_: mlproject.data.build_dataset
  dataset_name: food101
  data_dir: ${data_dir}
  sets_to_include: null
dataloader:
  _target_: torch.utils.data.dataloader.DataLoader
  dataset: null
  batch_size: ${train_batch_size}
  shuffle: true
  sampler: null
  batch_sampler: null
  num_workers: ${num_workers}
  collate_fn: null
  pin_memory: true
  drop_last: false
  timeout: 0.0
  worker_init_fn: null
  multiprocessing_context: null
  generator: null
  prefetch_factor: 2
  persistent_workers: false
  pin_memory_device: ''
optimizer:
  _target_: torch.optim.adamw.AdamW
  _partial_: true
  lr: 0.001
  betas:
  - 0.9
  - 0.999
  eps: 1.0e-08
  weight_decay: 0.01
  amsgrad: false
  maximize: false
  foreach: null
  capturable: false
scheduler:
  _target_: timm.scheduler.cosine_lr.CosineLRScheduler
  _partial_: true
  lr_min: 0.0
  cycle_mul: 1.0
  cycle_decay: 1.0
  cycle_limit: 1
  warmup_t: 0
  warmup_lr_init: 0
  warmup_prefix: false
  t_in_epochs: true
  noise_range_t: null
  noise_pct: 0.67
  noise_std: 1.0
  noise_seed: 42
  k_decay: 1.0
  initialize: true
learner:
  _target_: mlproject.boilerplate.Learner
  experiment_name: ${exp_name}
  experiment_dir: ${hf_repo_dir}
  model: null
  resume: ${resume}
  evaluate_every_n_steps: 500
  evaluate_every_n_epochs: null
  checkpoint_every_n_steps: 500
  checkpoint_after_validation: true
  train_iters: 10000
  train_epochs: null
  train_dataloader: null
  limit_train_iters: null
  val_dataloaders: null
  limit_val_iters: null
  test_dataloaders: null
  trainers: null
  evaluators: null
  callbacks: null
  print_model_parameters: false
callbacks:
  hf_uploader:
    _target_: mlproject.callbacks.UploadCheckpointsToHuggingFace
    repo_name: ${exp_name}
    repo_owner: ${hf_username}
wandb_args:
  _target_: wandb.sdk.wandb_init.init
  job_type: null
  dir: ${current_experiment_dir}
  config: null
  project: mlproject
  entity: null
  reinit: null
  tags: null
  group: null
  name: null
  notes: null
  magic: null
  config_exclude_keys: null
  config_include_keys: null
  anonymous: null
  mode: null
  allow_val_change: null
  resume: allow
  force: null
  tensorboard: null
  sync_tensorboard: null
  monitor_gym: null
  save_code: true
  id: null
  settings: null
hf_username: ???
seed: 42
freeze_backbone: false
resume: false
resume_from_checkpoint: null
print_config: false
train_batch_size: 125
eval_batch_size: 180
num_workers: 8
train: true
test: false
download_latest: true
download_checkpoint_with_name: null
root_experiment_dir: /experiments
data_dir: /data
current_experiment_dir: ${root_experiment_dir}/${exp_name}
repo_path: ${hf_username}/${exp_name}
hf_repo_dir: ${current_experiment_dir}/repo
code_dir: ${hydra:runtime.cwd}


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
</details>

To configure the compute optimizations use 
```bash
accelerate config
```
And answer the prompted questions.

Furthermore, instead of configuring the accelerate framework you can pass arguments directly, as explained when one issues:

```bash
accelerate launch --help
```
<details>
<summary>View default response</summary>

```bash
accelerate launch --help
usage: accelerate <command> [<args>] launch [-h] [--config_file CONFIG_FILE] [--cpu] [--mps] [--multi_gpu] [--tpu] [--use_mps_device]
    [--dynamo_backend {no,eager,aot_eager,inductor,nvfuser,aot_nvfuser,aot_cudagraphs,ofi,fx2trt,onnxrt,ipex}]
    [--mixed_precision {no,fp16,bf16}] [--fp16]
    [--num_processes NUM_PROCESSES] [--num_machines NUM_MACHINES]
    [--num_cpu_threads_per_process NUM_CPU_THREADS_PER_PROCESS]
    [--use_deepspeed] [--use_fsdp] [--use_megatron_lm]
    [--gpu_ids GPU_IDS] [--same_network] [--machine_rank MACHINE_RANK]
    [--main_process_ip MAIN_PROCESS_IP]
    [--main_process_port MAIN_PROCESS_PORT] [--rdzv_conf RDZV_CONF]
    [--max_restarts MAX_RESTARTS] [--monitor_interval MONITOR_INTERVAL]
    [-m] [--no_python] [--main_training_function MAIN_TRAINING_FUNCTION]
    [--downcast_bf16] [--deepspeed_config_file DEEPSPEED_CONFIG_FILE]
    [--zero_stage ZERO_STAGE]
    [--offload_optimizer_device OFFLOAD_OPTIMIZER_DEVICE]
    [--offload_param_device OFFLOAD_PARAM_DEVICE]
    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
    [--gradient_clipping GRADIENT_CLIPPING]
    [--zero3_init_flag ZERO3_INIT_FLAG]
    [--zero3_save_16bit_model ZERO3_SAVE_16BIT_MODEL]
    [--deepspeed_hostfile DEEPSPEED_HOSTFILE]
    [--deepspeed_exclusion_filter DEEPSPEED_EXCLUSION_FILTER]
    [--deepspeed_inclusion_filter DEEPSPEED_INCLUSION_FILTER]
    [--deepspeed_multinode_launcher DEEPSPEED_MULTINODE_LAUNCHER]
    [--fsdp_offload_params FSDP_OFFLOAD_PARAMS]
    [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS]
    [--fsdp_sharding_strategy FSDP_SHARDING_STRATEGY]
    [--fsdp_auto_wrap_policy FSDP_AUTO_WRAP_POLICY]
    [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
    [--fsdp_backward_prefetch_policy FSDP_BACKWARD_PREFETCH_POLICY]
    [--fsdp_state_dict_type FSDP_STATE_DICT_TYPE]
    [--megatron_lm_tp_degree MEGATRON_LM_TP_DEGREE]
    [--megatron_lm_pp_degree MEGATRON_LM_PP_DEGREE]
    [--megatron_lm_num_micro_batches MEGATRON_LM_NUM_MICRO_BATCHES]
    [--megatron_lm_sequence_parallelism MEGATRON_LM_SEQUENCE_PARALLELISM]
    [--megatron_lm_recompute_activations MEGATRON_LM_RECOMPUTE_ACTIVATIONS]
    [--megatron_lm_use_distributed_optimizer MEGATRON_LM_USE_DISTRIBUTED_OPTIMIZER]
    [--megatron_lm_gradient_clipping MEGATRON_LM_GRADIENT_CLIPPING]
    [--aws_access_key_id AWS_ACCESS_KEY_ID]
    [--aws_secret_access_key AWS_SECRET_ACCESS_KEY] [--debug]
    training_script ...

positional arguments:
  training_script       The full path to the script to be launched in parallel, followed by all the arguments for
                        the training script.
  training_script_args  Arguments of the training script.

options:
  -h, --help            Show this help message and exit.
  --config_file CONFIG_FILE
                        The config file to use for the default values in the launching script.
  -m, --module          Change each process to interpret the launch script as a Python module, executing with the
                        same behavior as 'python -m'.
  --no_python           Skip prepending the training script with 'python' - just execute it directly. Useful when
                        the script is not a Python script.
  --debug               Whether to print out the torch.distributed stack trace when something fails.

Hardware Selection Arguments:
  Arguments for selecting the hardware to be used.

  --cpu                 Whether or not to force the training on the CPU.
  --mps                 Whether or not this should use MPS-enabled GPU device on MacOS machines.
  --multi_gpu           Whether or not this should launch a distributed GPU training.
  --tpu                 Whether or not this should launch a TPU training.
  --use_mps_device      This argument is deprecated, use `--mps` instead.

Resource Selection Arguments:
  Arguments for fine-tuning how available hardware should be used.

  --dynamo_backend {no,eager,aot_eager,inductor,nvfuser,aot_nvfuser,aot_cudagraphs,ofi,fx2trt,onnxrt,ipex}
                        Choose a backend to optimize your training with dynamo, see more at
                        https://github.com/pytorch/torchdynamo.
  --mixed_precision {no,fp16,bf16}
                        Whether or not to use mixed precision training. Choose between FP16 and BF16 (bfloat16)
                        training. BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or
                        later.
  --fp16                This argument is deprecated, use `--mixed_precision fp16` instead.
  --num_processes NUM_PROCESSES
                        The total number of processes to be launched in parallel.
  --num_machines NUM_MACHINES
                        The total number of machines used in this training.
  --num_cpu_threads_per_process NUM_CPU_THREADS_PER_PROCESS
                        The number of CPU threads per process. Can be tuned for optimal performance.

Training Paradigm Arguments:
  Arguments for selecting which training paradigm to be used.

  --use_deepspeed       Whether to use deepspeed.
  --use_fsdp            Whether to use fsdp.
  --use_megatron_lm     Whether to use Megatron-LM.

Distributed GPUs:
  Arguments related to distributed GPU training.

  --gpu_ids GPU_IDS     What GPUs (by id) should be used for training on this machine as a comma-seperated list
  --same_network        Whether all machines used for multinode training exist on the same local network.
  --machine_rank MACHINE_RANK
                        The rank of the machine on which this script is launched.
  --main_process_ip MAIN_PROCESS_IP
                        The IP address of the machine of rank 0.
  --main_process_port MAIN_PROCESS_PORT
                        The port to use to communicate with the machine of rank 0.
  --rdzv_conf RDZV_CONF
                        Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).
  --max_restarts MAX_RESTARTS
                        Maximum number of worker group restarts before failing.
  --monitor_interval MONITOR_INTERVAL
                        Interval, in seconds, to monitor the state of workers.

TPU:
  Arguments related to TPU.

  --main_training_function MAIN_TRAINING_FUNCTION
                        The name of the main function to be executed in your script (only for TPU training).
  --downcast_bf16       Whether when using bf16 precision on TPUs if both float and double tensors are cast to
                        bfloat16 or if double tensors remain as float32.

DeepSpeed Arguments:
  Arguments related to DeepSpeed.

  --deepspeed_config_file DEEPSPEED_CONFIG_FILE
                        DeepSpeed config file.
  --zero_stage ZERO_STAGE
                        DeepSpeeds ZeRO optimization stage (useful only when `use_deepspeed` flag is passed).
  --offload_optimizer_device OFFLOAD_OPTIMIZER_DEVICE
                        Decides where (none|cpu|nvme) to offload optimizer states (useful only when
                        `use_deepspeed` flag is passed).
  --offload_param_device OFFLOAD_PARAM_DEVICE
                        Decides where (none|cpu|nvme) to offload parameters (useful only when `use_deepspeed`
                        flag is passed).
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        No of gradient_accumulation_steps used in your training script (useful only when
                        `use_deepspeed` flag is passed).
  --gradient_clipping GRADIENT_CLIPPING
                        gradient clipping value used in your training script (useful only when `use_deepspeed`
                        flag is passed).
  --zero3_init_flag ZERO3_INIT_FLAG
                        Decides Whether (true|false) to enable `deepspeed.zero.Init` for constructing massive
                        models. Only applicable with DeepSpeed ZeRO Stage-3.
  --zero3_save_16bit_model ZERO3_SAVE_16BIT_MODEL
                        Decides Whether (true|false) to save 16-bit model weights when using ZeRO Stage-3. Only
                        applicable with DeepSpeed ZeRO Stage-3.
  --deepspeed_hostfile DEEPSPEED_HOSTFILE
                        DeepSpeed hostfile for configuring multi-node compute resources.
  --deepspeed_exclusion_filter DEEPSPEED_EXCLUSION_FILTER
                        DeepSpeed exclusion filter string when using mutli-node setup.
  --deepspeed_inclusion_filter DEEPSPEED_INCLUSION_FILTER
                        DeepSpeed inclusion filter string when using mutli-node setup.
  --deepspeed_multinode_launcher DEEPSPEED_MULTINODE_LAUNCHER
                        DeepSpeed multi-node launcher to use.

FSDP Arguments:
  Arguments related to Fully Shared Data Parallelism.

  --fsdp_offload_params FSDP_OFFLOAD_PARAMS
                        Decides Whether (true|false) to offload parameters and gradients to CPU. (useful only
                        when `use_fsdp` flag is passed).
  --fsdp_min_num_params FSDP_MIN_NUM_PARAMS
                        FSDPs minimum number of parameters for Default Auto Wrapping. (useful only when
                        `use_fsdp` flag is passed).
  --fsdp_sharding_strategy FSDP_SHARDING_STRATEGY
                        FSDPs Sharding Strategy. (useful only when `use_fsdp` flag is passed).
  --fsdp_auto_wrap_policy FSDP_AUTO_WRAP_POLICY
                        FSDPs auto wrap policy. (useful only when `use_fsdp` flag is passed).
  --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        Transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`,
                        `T5Block` .... (useful only when `use_fsdp` flag is passed).
  --fsdp_backward_prefetch_policy FSDP_BACKWARD_PREFETCH_POLICY
                        FSDP's backward prefetch policy. (useful only when `use_fsdp` flag is passed).
  --fsdp_state_dict_type FSDP_STATE_DICT_TYPE
                        FSDP's state dict type. (useful only when `use_fsdp` flag is passed).

Megatron-LM Arguments:
  Arguments related to Megatron-LM.

  --megatron_lm_tp_degree MEGATRON_LM_TP_DEGREE
                        Megatron-LMs Tensor Parallelism (TP) degree. (useful only when `use_megatron_lm` flag is
                        passed).
  --megatron_lm_pp_degree MEGATRON_LM_PP_DEGREE
                        Megatron-LMs Pipeline Parallelism (PP) degree. (useful only when `use_megatron_lm` flag
                        is passed).
  --megatron_lm_num_micro_batches MEGATRON_LM_NUM_MICRO_BATCHES
                        Megatron-LMs number of micro batches when PP degree > 1. (useful only when
                        `use_megatron_lm` flag is passed).
  --megatron_lm_sequence_parallelism MEGATRON_LM_SEQUENCE_PARALLELISM
                        Decides Whether (true|false) to enable Sequence Parallelism when TP degree > 1. (useful
                        only when `use_megatron_lm` flag is passed).
  --megatron_lm_recompute_activations MEGATRON_LM_RECOMPUTE_ACTIVATIONS
                        Decides Whether (true|false) to enable Selective Activation Recomputation. (useful only
                        when `use_megatron_lm` flag is passed).
  --megatron_lm_use_distributed_optimizer MEGATRON_LM_USE_DISTRIBUTED_OPTIMIZER
                        Decides Whether (true|false) to use distributed optimizer which shards optimizer state
                        and gradients across Data Pralellel (DP) ranks. (useful only when `use_megatron_lm` flag
                        is passed).
  --megatron_lm_gradient_clipping MEGATRON_LM_GRADIENT_CLIPPING
                        Megatron-LMs gradient clipping value based on global L2 Norm (0 to disable). (useful
                        only when `use_megatron_lm` flag is passed).

AWS Arguments:
  Arguments related to AWS.

  --aws_access_key_id AWS_ACCESS_KEY_ID
                        The AWS_ACCESS_KEY_ID used to launch the Amazon SageMaker training job
  --aws_secret_access_key AWS_SECRET_ACCESS_KEY
                        The AWS_SECRET_ACCESS_KEY used to launch the Amazon SageMaker training job.
```
</details>

So, for example, to use bf16 mixed precision, one can do:

```bash
accelerate launch --mixed_precision=bf16 mlproject/run.py exp_name=test-bf16
```

## Modify the experiment configuration from the command line
Hydra-zen allows easy and quick configuration of your experiment via command line.

Three key cases are:

1. Set argument: 
```bash 
accelerate launch mlproject/run.py exp_name=my-awesome-experiment train_batch_size=50
```
2. Add a new argument not previously specified in the config:
```bash 
accelerate launch mlproject/run.py +my_new_argument=my_new_value
```
3. Remove an existing argument, previously specified in the config:
```bash 
accelerate launch mlproject/run.py ~train_batch_size
``` 

For more such syntax see the [hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/).

## Tracking experiments with wandb
The template supports wandb by default. So assuming you fill in the environment variable template with your wandb key and source the file as explained in the section **Setting up the relevant environment variables** wandb should be running. 

## Setting up the usage of huggingface model and dataset hubs so you can store your model weights and datasets
The template supports huggingface datasets and models by default. So assuming you fill in the environment variable template with your wandb key and source the file as explained in the section **Setting up the relevant environment variables**, you should be fine.

## Making any class and/or function configurable via Hydra-Zen
This template uses hydra-zen to grab any function or class and convert them into a configurable dataclass object that can then be accessed via the command line interface to modify an experiment configuration.

Furthermore, I have implemented a python decorator that can add a configuration generator function to a given class or function. More specifically, the [`configurable`](mlproject/decorators.py#L9) decorator. 

### Making a class or function configurable
To summarize, there are two different ways to make a class or function configurable:

1. Using the `configurable` decorator:
   
```python
@configurable
def build_something(batch_size: int, num_layers: int):
  return batch_size, num_layers

build_something_config = build_something.build_config(populate_full_signature=True)
```

where `build_something_config` is the config of the function `build_something`. More specifically, an instantiation of the config would look like this:

```python
print(build_something_config(batch_size=32, num_layers=2))
```

And the output will look like:
```bash
Builds_build_something(_target_='__main__.build_something', batch_size=32, num_layers=2)
```

This essentially shows us the target function for which the configuration parameters are being collected.

2. Using the builds function from hydra-zen:

```python
from hydra_zen import builds, instantiate
    
def build_something(batch_size: int, num_layers: int):
    return batch_size, num_layers

dummy_config = builds(build_something, populate_full_signature=True)

```

### Instantiating a function or class through its configuration object

 So one could then instantiate the function or class that the configuration has been built for, using:

```python
from hydra_zen import builds, instantiate

dummy_function_instantiation = instantiate(dummy_config)

print(dummy_function_instantiation)
```
which returns:

```bash
(32, 2)
```
Which is ofcourse the output of the function instantiation.

## Adding a new callback
The template has built in, a callback system which allows one to inject a small piece of code, referred to as a `callback` function at any stage of the training and evaluation process of their choosing. The reason for this is that it keeps the main boilerplate code clean and tidy, while allowing the flexibility of adding whatever functions one needs at any point in the training. 

All the possible entry points can be found in the [callbacks module](mlproject/callbacks.py#L28), as well as the available/exposed data items and experiment variables that the functions can use. 

So, when one wants to build a new function, they need to inherit from the `Callback` class and then implement one or more of the signature methods. For an example look at the [`UploadCheckpointsToHuggingFace`](mlproject/callbacks.py#L407) callback.

## Adding a new model
To add a new model simply modify the existing build_model function found in [models.py](mlproject/models.py#L18), or simply find the model you need from the HuggingFace model repository and add the relevant classes and model name to the build model function.

## Adding a new dataset
To add a new dataset simply modify the existing build_dataset function found in [data.py](mlproject/data.py#L9), or simply find the model you need from the HuggingFace dataset library and add the relevant classes and dataset name to the build dataset function.


## Running a kubernetes hyperparameter search

TODO: Show a small tutorial on how to run a kubernetes hyperparameter search using the framework. 


References: 

@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}

@article{soklaski2022tools,
  title={Tools and Practices for Responsible AI Engineering},
  author={Soklaski, Ryan and Goodwin, Justin and Brown, Olivia and Yee, Michael and Matterer, Jason},
  journal={arXiv preprint arXiv:2201.05647},
  year={2022}
}





