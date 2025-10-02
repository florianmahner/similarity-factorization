"""Base dataset result container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ndarray = np.ndarray


@dataclass
class DatasetResult:
    """
    Standard container for all dataset loaders.

    All loaders return an instance of this class, ensuring a consistent interface
    across different datasets.

    Parameters
    ----------
    name : str
        Dataset name
    data : ndarray or dict, optional
        Raw features or activations. Can be dict for multiple feature types.
    rsm : ndarray, optional
        Similarity matrix (n_samples, n_samples)
    targets : ndarray, optional
        Labels or ground truth
    metadata : dict, optional
        Additional dataset-specific data

    Attributes
    ----------
    name : str
        Dataset identifier
    data : ndarray or dict or None
        Feature matrix or dict of features
    rsm : ndarray or None
        Representational similarity matrix
    targets : ndarray or None
        Target labels or values
    metadata : dict
        Extra dataset-specific attributes

    Examples
    --------
    >>> result = DatasetResult(name='iris', data=X, targets=y, rsm=rsm)
    >>> result.name
    'iris'
    >>> result.data.shape
    (150, 4)
    """

    name: str
    data: ndarray | dict | None = None
    rsm: ndarray | None = None
    targets: ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __getattr__(self, key: str):
        """Allow access to metadata attributes."""
        if "metadata" in self.__dict__ and key in self.metadata:
            return self.metadata[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def keys(self) -> list[str]:
        """List available attributes."""
        base_keys = ["name"]
        if self.data is not None:
            base_keys.append("data")
        if self.rsm is not None:
            base_keys.append("rsm")
        if self.targets is not None:
            base_keys.append("targets")
        return base_keys + list(self.metadata.keys())

    def summary(self) -> None:
        """Print dataset summary."""
        print(f"Dataset: {self.name}")
        if self.data is not None:
            if isinstance(self.data, dict):
                print(f"  data: {len(self.data)} feature types")
                for k, v in self.data.items():
                    print(f"    {k}: {v.shape}")
            else:
                print(f"  data: {self.data.shape}")
        if self.rsm is not None:
            print(f"  rsm: {self.rsm.shape}")
        if self.targets is not None:
            print(f"  targets: {self.targets.shape}")
        if self.metadata:
            print(f"  metadata: {list(self.metadata.keys())}")
