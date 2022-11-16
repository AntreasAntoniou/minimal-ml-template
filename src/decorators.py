import functools
from typing import Callable
from hydra_zen import builds, instantiate
from prometheus_client import Metric


def configurable(func: Callable) -> Callable:
    func.__configurable__ = True
    func.default_config = builds(func, populate_full_signature=True)
    return func


def check_if_configurable(func: Callable, phase_name: str) -> bool:
    return func.__configurable__ if hasattr(func, "__configurable__") else False


def collect_metrics(func: Callable) -> Callable:
    def collect_metrics(pl_module, metrics_dict, phase_name) -> None:
        for metric_key, computed_value in metrics_dict.items():

            if computed_value is not None:
                pl_module.log(
                    name=f"{phase_name}/{metric_key}",
                    value=computed_value.detach(),
                    prog_bar="opt_loss" in metric_key,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )

    @functools.wraps(func)
    def wrapper_collect_metrics(*args, **kwargs):
        pl_module = args[0]
        outputs = func(*args, **kwargs)
        metrics_dict = outputs["metrics"]
        phase_name = outputs["phase_name"]
        collect_metrics(
            pl_module=pl_module, metrics_dict=metrics_dict, phase_name=phase_name
        )
        return outputs

    return wrapper_collect_metrics


def test_wrapper(phase_name: str):
    def wrapper(func):
        print(f"test_wrapper {phase_name}")
        return func

    return wrapper


if __name__ == "__main__":

    @test_wrapper(phase_name="train")
    def test_method(input_dict: dict):
        for key, value in input_dict.items():
            print(f"{key}: {value}")

    # @configurable
    # class DummyObject(object):
    #     def __init__(self, weight_decay: float, random_crap: str):
    #         self.weight_decay = weight_decay
    #         self.random_crap = random_crap

    #     def __repr__(self) -> str:
    #         return f"DummyObject(weight_decay={self.weight_decay}, random_crap={self.random_crap})"

    # module = DummyObject(0.1, "hello")

    # print(check_if_configurable(module))

    # module_from_config = instantiate(
    #     module.default_config, weight_decay=0.2, random_crap="world"
    # )

    # print(check_if_configurable(module_from_config), module_from_config)
