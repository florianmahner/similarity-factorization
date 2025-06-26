from abc import ABC, abstractmethod
from .cfg import get_dataset_path


class BaseDataset:
    """Dataset container that automatically assigns keyword arguments as attributes."""

    def __init__(self, **kwargs):
        """Store all dataset components as attributes dynamically."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def keys(self):
        """List available dataset components."""
        return list(self.__dict__.keys())

    def summary(self):
        """Print a structured overview of the dataset components."""
        print("Dataset Summary:")
        for key, value in self.__dict__.items():
            if key == "metadata":
                print(f"  - {key}: {len(value)} metadata entries")
            elif hasattr(value, "shape"):
                print(f"  - {key}: shape {value.shape}")
            elif isinstance(value, list):
                print(f"  - {key}: {len(value)} items")
            else:
                print(f"  - {key}: {type(value)}")


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders, ensuring they return `BaseDataset`."""

    def __init__(self, name: str, root: str = None, **kwargs):
        self.name = name
        self.root = root or get_dataset_path(name)  # Default dataset path
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> BaseDataset:
        """All dataset loaders must implement this method and return a `BaseDataset`."""
        pass

    def summary(self):
        """Print a structured overview of the dataset."""
        dataset = self.load()
        dataset.summary()
