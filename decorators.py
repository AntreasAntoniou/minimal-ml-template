import functools
from typing import Callable
from hydra_zen import builds, instantiate


def configurable(func: Callable) -> Callable:
    func.__configurable__ = True
    func.default_config = builds(func, populate_full_signature=True)
    return func


def check_if_configurable(func: Callable) -> bool:
    return func.__configurable__ if hasattr(func, "__configurable__") else False


if __name__ == "__main__":

    @configurable
    class DummyObject(object):
        def __init__(self, weight_decay: float, random_crap: str):
            self.weight_decay = weight_decay
            self.random_crap = random_crap

        def __repr__(self) -> str:
            return f"DummyObject(weight_decay={self.weight_decay}, random_crap={self.random_crap})"

    module = DummyObject(0.1, "hello")

    print(check_if_configurable(module))

    module_from_config = instantiate(
        module.default_config, weight_decay=0.2, random_crap="world"
    )

    print(check_if_configurable(module_from_config), module_from_config)
