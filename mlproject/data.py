from datasets import load_dataset

from mlproject.decorators import configurable


@configurable
def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include=None,
):
    if sets_to_include is None:
        sets_to_include = ["train", "validation"]

    dataset = {}
    for set_name in sets_to_include:
        data = load_dataset(
            path=dataset_name,
            split=set_name,
            cache_dir=data_dir,
            task="image-classification",
        )
        dataset[set_name] = data

    return dataset
