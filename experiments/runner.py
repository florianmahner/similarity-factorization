#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import tomlparse
import tomllib

from cli.registry import get_experiment, list_experiments
from cli.context import (
    ExperimentContext,
    make_run_dir,
    make_logger,
    write_metadata,
)


def discover_experiments(repo_root: Path) -> None:
    exp_root = repo_root / "experiments"
    candidates: list[Path] = []
    dev = exp_root / "development"
    if dev.exists():
        for p in dev.iterdir():
            ep = p / "exp.py"
            if ep.exists():
                candidates.append(ep)
    for p in exp_root.iterdir():
        if p.name == "development":
            continue
        ep = p / "exp.py"
        if ep.exists():
            candidates.append(ep)
    for path in candidates:
        # Use proper package name instead of synthetic name
        relative_path = path.parent.relative_to(repo_root)
        module_name = str(relative_path).replace('/', '.') + '.exp'

        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        # Set __package__ to enable relative imports
        module.__package__ = '.'.join(module_name.split('.')[:-1])
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            experiment_name = path.parent.name
            print(f"Warning: Failed to load experiment '{experiment_name}': {e}", file=sys.stderr)
            continue


def build_bootstrap_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--experiment", "-e")
    p.add_argument("--mode", "-m", choices=["dev", "stable"], default="dev")
    p.add_argument("--toml", type=Path, default=Path("config/experiments.toml"))
    p.add_argument("--table")
    p.add_argument("--root-table")
    p.add_argument("--name", "-n")
    p.add_argument("--list", action="store_true")
    p.add_argument("--print-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def args_to_params_dict(params_type: type, ns: argparse.Namespace) -> dict:
    if not is_dataclass(params_type):
        return vars(ns)
    allowed = {f.name for f in fields(params_type)}
    return {k: getattr(ns, k) for k in allowed if hasattr(ns, k)}


def load_toml_data(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        return {}


def extract_task_overrides(
    data: dict[str, Any], experiment: str, mode: str, task: str | None
) -> dict[str, Any]:
    if not task:
        return {}
    base = data.get(experiment, {})
    if not isinstance(base, dict):
        return {}

    overrides: dict[str, Any] = {}
    base_table = base.get(task)
    if isinstance(base_table, dict):
        overrides.update(
            {k: v for k, v in base_table.items() if not isinstance(v, dict)}
        )
    if mode == "dev":
        dev_table = base.get("dev")
        if isinstance(dev_table, dict):
            nested = dev_table.get(task)
            if isinstance(nested, dict):
                overrides.update(
                    {k: v for k, v in nested.items() if not isinstance(v, dict)}
                )
    return overrides


def apply_task_overrides(
    params_dict: dict[str, Any],
    overrides: dict[str, Any],
    params_type: type,
) -> dict[str, Any]:
    if not overrides or not is_dataclass(params_type):
        return params_dict
    try:
        defaults = params_type()
    except TypeError:
        return params_dict

    for key, value in overrides.items():
        if key not in params_dict:
            continue
        current = params_dict[key]
        default_value = getattr(defaults, key, object())
        if current == default_value:
            params_dict[key] = value
    return params_dict


def main():
    repo_root = Path(__file__).resolve().parents[1]
    discover_experiments(repo_root)

    boot = build_bootstrap_parser().parse_known_args()[0]
    if boot.list:
        for name in list_experiments():
            print(name)
        return
    if not boot.experiment:
        raise SystemExit("--experiment required (use --list to see options)")

    spec = get_experiment(boot.experiment)

    parser = tomlparse.ArgumentParser(description="Unified experiments runner")
    parser.add_argument("--experiment", "-e", required=True)
    parser.add_argument("--mode", "-m", choices=["dev", "stable"], default=boot.mode)
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Output directory name for stable runs (defaults to task name if not specified)",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    if spec.add_arguments is not None:
        spec.add_arguments(parser)

    parser.set_defaults(toml=boot.toml)

    # Set TOML table defaults based on mode
    if not boot.root_table:
        parser.set_defaults(root_table=boot.experiment)

    if not boot.table:
        # In dev mode: use experiment.dev as override table
        # In stable mode: just use experiment as base table
        if boot.mode == "dev":
            parser.set_defaults(table=f"{boot.experiment}.dev")
        else:
            parser.set_defaults(table=boot.experiment)

    args = parser.parse_args()

    # Get task name from args if available
    task_name = getattr(args, "task", None)
    run_dir = make_run_dir(args.experiment, args.mode, repo_root, args.name, task_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(run_dir)
    rng = np.random.default_rng(getattr(args, "seed", 0))

    if args.print_config:
        resolved = {
            k: v
            for k, v in vars(args).items()
            if k not in {"toml", "table", "root_table", "print_config", "dry_run"}
        }
        logger.info("Resolved arguments: %s", resolved)
        return
    if args.dry_run:
        logger.info("Dry run: %s in %s at %s", args.experiment, args.mode, run_dir)
        return

    ctx = ExperimentContext(
        experiment=args.experiment,
        mode=args.mode,
        run_dir=run_dir,
        logger=logger,
        rng=rng,
    )
    write_metadata(
        run_dir,
        {
            "experiment": args.experiment,
            "mode": args.mode,
            "toml": getattr(args, "toml", None),
            "table": getattr(args, "table", None),
            "root_table": getattr(args, "root_table", None),
            "run_dir": str(run_dir),
        },
    )

    params_dict = args_to_params_dict(spec.params_type, args)
    config_path = Path(getattr(args, "toml", parser.get_default("toml")))
    config_data = load_toml_data(config_path)
    overrides = extract_task_overrides(
        config_data,
        args.experiment,
        args.mode,
        params_dict.get("task"),
    )
    params_dict = apply_task_overrides(params_dict, overrides, spec.params_type)
    params = (
        spec.params_type(**params_dict)
        if is_dataclass(spec.params_type)
        else params_dict
    )

    spec.run(ctx, params)


if __name__ == "__main__":
    main()
