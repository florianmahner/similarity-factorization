from sklearn.datasets import (
    load_iris,
    load_digits,
    load_diabetes,
    load_wine,
    load_breast_cancer,
)

from scipy.io import loadmat
from torchvision.datasets import MNIST as MNIST_torch
from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification


@register_dataset("iris")
class Iris(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("iris", root)

    def load(self) -> BaseDataset:
        iris = load_iris()
        X = iris.data
        y = iris.target
        rsm = np.corrcoef(X)
        return BaseDataset(
            name="iris",
            data=X,
            targets=y,
            rsm=rsm,
        )


@register_dataset("diabetes")
class Diabetes(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("diabetes", root)

    def load(self) -> BaseDataset:
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.data.to_numpy()
        y = diabetes.target.to_numpy()
        return BaseDataset(name="diabetes", data=X, targets=y)


@register_dataset("digits")
class Digits(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("digits", root)

    def load(self) -> BaseDataset:
        digits = load_digits()
        return BaseDataset(
            name="digits",
            data=digits.data,
            targets=digits.target,
        )


@register_dataset("mnist")
class MNIST(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("mnist", root)

    def load(self) -> BaseDataset:
        train = MNIST_torch(self.root, train=True, download=True)
        test = MNIST_torch(self.root, train=False, download=True)

        return BaseDataset(
            name="mnist",
            train_data=train.data.numpy(),
            train_targets=train.targets.numpy(),
            test_data=test.data.numpy(),
            test_targets=test.targets.numpy(),
        )


@register_dataset("orl")
class ORL(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("orl", root)

    def load(self) -> BaseDataset:
        root = Path(self.root)
        file = root / "ORL.mat"
        if not file.exists():
            raise FileNotFoundError(f"File {file} not found")
        data = loadmat(file)
        x = data["data"]
        y = data["label"].squeeze()
        return BaseDataset(name="orl", data=x, targets=y)


@register_dataset("synthetic")
class SyntheticDataset(BaseDatasetLoader):
    def __init__(
        self,
        n_classes: int = 4,
        n_features: int = 15,
        samples_per_class: int = 100,
        flip_y: float = 0.0,
        class_sep: float = 1.0,
        random_state: int | None = None,
    ):
        super().__init__("synthetic", "./")
        self.n_classes = n_classes
        self.n_features = n_features
        self.samples_per_class = samples_per_class
        self.flip_y = flip_y
        self.class_sep = class_sep
        self.random_state = random_state

    def load(self) -> BaseDataset:
        x, y = make_classification(
            n_samples=self.samples_per_class * self.n_classes,
            n_features=self.n_features,
            n_informative=self.n_classes,
            n_redundant=1,
            n_clusters_per_class=1,
            n_classes=self.n_classes,
            random_state=self.random_state,
            class_sep=self.class_sep,
        )
        return BaseDataset(name="synthetic", data=x, targets=y)


@register_dataset("wine")
class Wine(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("wine", root)

    def load(self) -> BaseDataset:
        wine = load_wine()
        X = wine.data
        y = wine.target
        rsm = np.corrcoef(X)
        return BaseDataset(
            name="wine",
            data=X,
            targets=y,
            rsm=rsm,
        )


@register_dataset("breast_cancer")
class BreastCancer(BaseDatasetLoader):
    def __init__(self, root: str = None):
        super().__init__("breast_cancer", root)

    def load(self) -> BaseDataset:
        cancer = load_breast_cancer()
        X = cancer.data
        y = cancer.target
        return BaseDataset(
            name="breast_cancer",
            data=X,
            targets=y,
        )
