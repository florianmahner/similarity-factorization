from .link_prediction import run as run_link_prediction
from .validate_corum import run as run_corum_validation
from .node_classification import run as run_node_classification

__all__ = ["run_link_prediction", "run_corum_validation", "run_node_classification"]
