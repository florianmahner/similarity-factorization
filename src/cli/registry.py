from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ExperimentContext


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    params_type: type
    add_arguments: Callable[[Any], None] | None
    run: Callable[["ExperimentContext", Any], None]
    description: str | None = None


_registry: dict[str, Experiment] = {}


def register_experiment(
    name: str,
    params_type: type,
    add_arguments: Callable[[Any], None] | None = None,
    description: str | None = None,
):
    def decorator(run: Callable[["ExperimentContext", Any], None]):
        if name in _registry:
            raise ValueError(f"Experiment already registered: {name}")
        _registry[name] = Experiment(
            name=name,
            params_type=params_type,
            add_arguments=add_arguments,
            run=run,
            description=description,
        )
        return run

    return decorator


def get_experiment(name: str) -> Experiment:
    try:
        return _registry[name]
    except KeyError as e:
        available = ", ".join(sorted(_registry))
        raise KeyError(f"Unknown experiment: {name}. Available: {available}") from e


def list_experiments() -> list[str]:
    return sorted(_registry)
