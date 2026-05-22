"""Config loading: YAML defaults with dotted-key CLI overrides."""
from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load YAML config and apply dotted-key overrides (e.g. 'schedule.epochs').

    Overrides whose value is None are ignored (lets argparse defaults pass through
    without clobbering the YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_dotted(cfg, dotted, value)
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def resolve_dim_weights(
    nesting_dims: list[int],
    weighting: str,
    explicit: list[float] | None,
) -> list[float]:
    """Turn a weighting scheme into concrete per-dim weights.

    explicit, if given, wins and must match len(nesting_dims).
    'uniform' -> all 1.0. 'increasing' -> proportional to dim, normalized so the
    smallest dim has weight 1.0."""
    if explicit is not None:
        if len(explicit) != len(nesting_dims):
            raise ValueError(
                f"dim_weights length {len(explicit)} != nesting_dims length {len(nesting_dims)}"
            )
        return [float(w) for w in explicit]
    if weighting == "uniform":
        return [1.0] * len(nesting_dims)
    if weighting == "increasing":
        smallest = min(nesting_dims)
        return [d / smallest for d in nesting_dims]
    raise ValueError(f"unknown weighting scheme: {weighting!r}")
