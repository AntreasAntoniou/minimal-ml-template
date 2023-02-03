import copy
from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor

from mlproject.decorators import configurable


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any


@configurable
def build_model(
    model_name: str = "google/vit-base-patch16-224-in21k",
    pretrained: bool = True,
    num_classes: int = 100,
):
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model: nn.Module = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_classes
    )

    if not pretrained:
        model.init_weights()

    transform = lambda image: feature_extractor(
        images=image, return_tensors="pt"
    )

    class Convert1ChannelTo3Channel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            temp = None
            if hasattr(x, "pixel_values"):
                temp = copy.copy(x)
                x = x["pixel_values"]
            x = ToTensor()(x)
            if len(x.shape) == 3 and x.shape[0] == 1:
                x = x.repeat([3, 1, 1])
            elif len(x.shape) == 4 and x.shape[1] == 1:
                x = x.repeat([1, 3, 1, 1])

            if temp is not None:
                temp["pixel_values"] = x
                x = temp

            return x

    pre_transform = Convert1ChannelTo3Channel()

    def transform_wrapper(input_dict: Dict):
        input_dict["image"][0] = pre_transform(input_dict["image"][0])

        return {
            "pixel_values": transform(input_dict["image"])["pixel_values"],
            "labels": input_dict["labels"],
        }

    return ModelAndTransform(model=model, transform=transform_wrapper)
