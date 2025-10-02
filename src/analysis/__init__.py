# src/experiments/__init__.py
_EXPERIMENTS = {}


def register_experiment(name: str):
    """Simple decorator to register experiment functions"""

    def decorator(func):
        _EXPERIMENTS[name] = func
        return func

    return decorator


def get_experiment(name: str):
    """Get experiment function by name"""
    return _EXPERIMENTS[name]


def list_experiments():
    """List all registered experiments"""
    return list(_EXPERIMENTS.keys())


def run_experiment(name: str, **kwargs):
    """Run any experiment by name"""
    return _EXPERIMENTS[name](**kwargs)
