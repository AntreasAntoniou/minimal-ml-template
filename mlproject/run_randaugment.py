import copy
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tabnanny import check
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from datasets import Image, load_dataset
from rich import print
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, Compose, Resize, ToTensor

from mlproject.callbacks import Callback, CallbackHandler, Interval
from mlproject.evaluators import ClassificationEvaluator, Evaluator
from mlproject.trainers import ClassificationTrainer, Trainer
from mlproject.utils import get_logger

logger = get_logger(__name__)