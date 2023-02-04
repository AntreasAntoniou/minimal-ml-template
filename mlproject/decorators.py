import functools
from functools import wraps
from typing import Any, Callable

import torch
from hydra_zen import builds, instantiate


def configurable(func: Callable) -> Callable:
    func.__configurable__ = True

    def build_config(**kwargs):
        return builds(func, **kwargs)

    setattr(func, "build_config", build_config)
    return func


def check_if_configurable(func: Callable, phase_name: str) -> bool:
    return (
        func.__configurable__ if hasattr(func, "__configurable__") else False
    )


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(
        step_idx: int,
        metrics_dict: dict(),
        phase_name: str,
        experiment_tracker: Any,
    ) -> None:
        for metric_key, computed_value in metrics_dict.items():
            if computed_value is not None:
                value = (
                    computed_value.detach()
                    if isinstance(computed_value, torch.Tensor)
                    else computed_value
                )
                experiment_tracker.log(
                    {f"{phase_name}/{metric_key}": value},
                    step=step_idx,
                )

                # print(f"{phase_name}/{metric_key} {value} {step_idx}")

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        outputs = func(*args, **kwargs)
        experiment_tracker = args[0].experiment_tracker
        metrics_dict = outputs.metrics
        phase_name = outputs.phase_name
        step_idx = outputs.step_idx
        collect_metrics(
            step_idx=step_idx,
            metrics_dict=metrics_dict,
            phase_name=phase_name,
            experiment_tracker=experiment_tracker,
        )
        return outputs

    return wrapper_collect_metrics


if __name__ == "__main__":

    @configurable
    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    build_something_config = build_something.build_config(
        populate_full_signature=True
    )
    dummy_config = build_something_config(batch_size=32, num_layers=2)
    print(dummy_config)

    from hydra_zen import builds, instantiate

    def build_something(batch_size: int, num_layers: int):
        return batch_size, num_layers

    dummy_config = builds(build_something, populate_full_signature=True)

    dummy_function_instantiation = instantiate(dummy_config)

    print(dummy_function_instantiation)
