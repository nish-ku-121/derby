#!/usr/bin/env python3
"""Generate a directory of concrete experiment configs from a sweep spec.

Minimal spec format (YAML):

sweep_name: reinforce_unified_demo          # required
base_config: configs/one_camp_n_days_v2_config.yaml
override:                                    # (optional) static dotted-key overrides applied before grid
  agents.0.policy: REINFORCE
  num_epochs: 3
  num_trajs: 80
grid:                                        # (optional) cartesian parameter grid (dotted keys → list)
  agents.0.params.actor_final_activation: [softplus, relu]
  agents.0.params.actor_hidden_layers: [1, 4]
  agents.0.params.actor_hidden_units: [1, 6]
  agents.0.params.actor_hidden_activation: [leaky_relu, elu]
restrict:                                    # (optional)
  max_combinations: 64
  sample_fraction: 0.5
label_template: "{actor_final_activation}|L={actor_hidden_layers}|U={actor_hidden_units}|ah={actor_hidden_activation}"  # (optional)

Outputs: sweeps/<sweep_name>/configs/run_0001.yaml ...
Each config's 'label' field set to: <sweep_name>__<suffix> (suffix derived from template or canonical param list)

Usage:
  python -m pipeline.make_grid --spec sweepspec.yaml
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("PyYAML required. Install via 'poetry add pyyaml'", file=sys.stderr)
    raise


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _write_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split('.')
    cur: Any = cfg
    for i, part in enumerate(parts):
        last = (i == len(parts) - 1)
        if part.isdigit():
            idx = int(part)
            if not isinstance(cur, list):
                raise TypeError(f"Expected list at segment '{part}' in path '{dotted_key}'")
            if idx >= len(cur):
                raise IndexError(f"Index {idx} out of range for path '{dotted_key}'")
            if last:
                cur[idx] = value
            else:
                cur = cur[idx]
        else:
            if last:
                cur[part] = value
            else:
                if part not in cur or not isinstance(cur[part], (dict, list)):
                    # create intermediate dict (lists must already exist via indices)
                    cur[part] = {}
                cur = cur[part]


def _extract_label_vars(dotted_keys: List[str]) -> List[str]:
    # Use last token (after final dot) as the variable name for label template context
    names = [k.split('.')[-1] for k in dotted_keys]
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    # Deterministic ordering: sort keys so numbering stable across runs
    keys.sort()
    product = itertools.product(*(grid[k] for k in keys))
    combos: List[Dict[str, Any]] = []
    for values in product:
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for k, v in overrides.items():
        _set_by_dotted_key(cfg, k, v)


def _sample_combos(combos: List[Dict[str, Any]], max_combos: int | None, sample_fraction: float | None, seed: int | None) -> List[Dict[str, Any]]:
    out = combos
    if max_combos is not None and max_combos >= 0:
        out = out[:max_combos]
    if sample_fraction is not None:
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be in (0,1]")
        rng = random.Random(seed)
        n = max(1, int(len(out) * sample_fraction))
        out = rng.sample(out, n)
        # Re-sort by original key order (not strictly necessary but keeps numbering deterministic-ish)
        out = sorted(out, key=lambda d: tuple(str(d[k]) for k in sorted(d.keys())))
    return out


def build_label(base_name: str, combo: Dict[str, Any], template: str | None) -> str:
    if template:
        # Build context with last-segment names; if collisions occur, later keys override earlier
        ctx: Dict[str, Any] = {}
        for full_key, value in combo.items():
            ctx[full_key.split('.')[-1]] = value
        try:
            suffix = template.format(**ctx)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"label_template references missing key '{missing}'. Available: {list(ctx)}") from e
    else:
        # Canonical key=value|... using short names
        parts = []
        for fk in sorted(combo.keys()):
            parts.append(f"{fk.split('.')[-1]}={combo[fk]}")
        suffix = "|".join(parts) if parts else "base"
    return f"{base_name}__{suffix}"


def generate_configs(spec_path: str, output_root: str | None = None) -> None:
    spec = _read_yaml(spec_path)
    sweep_name = spec.get('sweep_name')
    if not sweep_name:
        raise ValueError("sweep_name is required in sweep spec")
    base_config_path = spec.get('base_config')
    if not base_config_path:
        raise ValueError("base_config is required in sweep spec")
    base_cfg = _read_yaml(base_config_path)

    overrides: Dict[str, Any] = spec.get('override', {}) or {}
    grid: Dict[str, List[Any]] = spec.get('grid', {}) or {}
    restrict: Dict[str, Any] = spec.get('restrict', {}) or {}
    label_template: str | None = spec.get('label_template')
    seed = spec.get('seed')

    combos = expand_grid(grid)
    combos = _sample_combos(
        combos,
        max_combos=restrict.get('max_combinations'),
        sample_fraction=restrict.get('sample_fraction'),
        seed=seed,
    )

    # Prepare output directory
    root = Path(output_root) if output_root else Path("sweeps")
    out_dir = root / sweep_name / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply overrides once to a fresh copy per combo
    written = 0
    for idx, combo in enumerate(combos, start=1):
        cfg = deepcopy(base_cfg)
        apply_overrides(cfg, overrides)
        for k, v in combo.items():
            apply_overrides(cfg, {k: v})
        # Label update
        label = build_label(sweep_name, combo, label_template)
        cfg['label'] = label
        # Ensure each agent has consistent policy params section if present
        # (We don't mutate beyond provided dotted keys.)
        run_path = out_dir / f"run_{idx:04d}.yaml"
        _write_yaml(run_path, cfg)
        written += 1

    print(f"[make_grid] Wrote {written} configs to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate concrete configs from a sweep spec")
    ap.add_argument('--spec', required=True, help='Path to sweep spec YAML')
    ap.add_argument('--output-root', help='Optional root directory (default sweeps/)')
    args = ap.parse_args()
    generate_configs(args.spec, args.output_root)


if __name__ == '__main__':
    main()
