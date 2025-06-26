from typing import Type
from .base import BaseDataset, BaseDatasetLoader
import inspect

DATASET_REGISTRY = {}


def register_dataset(name: str):
    """Decorator to register dataset classes in the global registry."""

    def decorator(cls: Type[BaseDatasetLoader]):
        if not issubclass(cls, BaseDatasetLoader):
            raise TypeError(
                f"Registered dataset '{name}' must inherit from BaseDatasetLoader"
            )
        DATASET_REGISTRY[name] = cls  # Store the class, not an instance
        return cls

    return decorator


def load_dataset(name: str, **kwargs) -> BaseDataset:
    """Retrieve dataset instance from the registry and call its `load()` method."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not registered. Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls = DATASET_REGISTRY[name]

    # Get __init__ signature and filter kwargs to only pass valid ones
    init_signature = inspect.signature(dataset_cls.__init__)

    # Check if the function accepts **kwargs (VAR_KEYWORD parameter)
    accepts_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in init_signature.parameters.values()
    )

    if accepts_var_keyword:
        # If function accepts **kwargs, pass all kwargs
        valid_kwargs = kwargs
    else:
        # Otherwise, filter to only valid parameter names
        valid_kwargs = {
            k: v for k, v in kwargs.items() if k in init_signature.parameters
        }

    # Instantiate dataset loader
    dataset_loader = dataset_cls(**valid_kwargs)

    dataset = dataset_loader.load()

    if not isinstance(dataset, BaseDataset):
        raise TypeError(
            f"Dataset loader '{name}' returned {type(dataset)} instead of `BaseDataset`."
        )

    return dataset
