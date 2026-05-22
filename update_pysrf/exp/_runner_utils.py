"""Small helper used by tutorials and configuration-driven runs.

Provides `load_config(name)` which returns the JSON-parsed config dict for one
of the registered configs under RECIPE_K/configs/.  Experiment scripts in this
directory currently hard-code their values from the JSON files (they predate
this helper); this module exists so a notebook or new script can read the same
defaults without duplicating them.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CONFIG_DIR = os.path.join(_ROOT, "configs")
DATA_DIR = os.path.join(_ROOT, "data")
OUTPUT_DIR = os.path.join(_ROOT, "output")
SRC_DIR = os.path.join(_ROOT, "src")


def load_config(name):
    """Load one of the registered configs by file name (with or without .json)."""
    if not name.endswith(".json"):
        name = name + ".json"
    path = os.path.join(CONFIG_DIR, name)
    with open(path) as f:
        return json.load(f)


def add_recipe_k_src_to_path():
    """Insert RECIPE_K/src/ at the head of sys.path (idempotent)."""
    import sys
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
