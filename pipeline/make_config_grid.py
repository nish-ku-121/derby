#!/usr/bin/env python3
"""Generate concrete experiment configs from a sweep spec.

Minimal spec keys:
    sweep_name (str, required)
    base_config (path, required)
    override (mapping, optional) - dotted-key overrides
    grid (mapping, optional) - dotted-key -> list for cartesian product
    restrict.max_combinations (optional)

Agent labels may be literal strings or templates using generated-config dotted
paths, for example:
    agents.0.label: "REINFORCE|LR={agents.0.params.learning_rate}"

Example:
    sweep_name: demo
    base_config: configs/one_camp_n_days_base.yaml
    override:
        agents.0.policy: REINFORCE
    grid:
        agents.0.params.actor_hidden_layers: [1, 3]
        agents.0.params.actor_hidden_units: [8, 32]

Outputs placed in: sweeps/<sweep_name>/configs/run_0001.yaml etc.
"""
from __future__ import annotations

import argparse
import itertools
import string
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict, List


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover
        print("PyYAML required. Install via 'poetry add pyyaml'", file=sys.stderr)
        raise
    return yaml


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    yaml = _load_yaml_module()
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _write_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    yaml = _load_yaml_module()
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


def _restrict_combos(combos: List[Dict[str, Any]], max_combos: int | None) -> List[Dict[str, Any]]:
    out = combos
    if max_combos is not None and max_combos >= 0:
        out = out[:max_combos]
    return out


def _get_by_dotted_key(cfg: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = cfg
    for part in dotted_key.split('.'):
        if part.isdigit():
            if not isinstance(cur, list):
                raise KeyError(dotted_key)
            idx = int(part)
            if idx >= len(cur):
                raise KeyError(dotted_key)
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(dotted_key)
            cur = cur[part]
    return cur


def _render_agent_label_template(template: str, cfg: Dict[str, Any]) -> str:
    formatter = string.Formatter()
    parts: List[str] = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        parts.append(literal_text)
        if field_name is None:
            continue
        if not field_name:
            raise ValueError("agent label templates do not support anonymous placeholders")
        try:
            value = _get_by_dotted_key(cfg, field_name)
        except KeyError as e:
            raise ValueError(
                f"agent label template references missing config path '{field_name}'. "
                "Use full generated-config paths such as "
                "{agents.0.params.dist_type} or add the value to the base config/grid."
            ) from e
        if conversion == 'r':
            value = repr(value)
        elif conversion == 's':
            value = str(value)
        elif conversion == 'a':
            value = ascii(value)
        elif conversion is not None:
            raise ValueError(f"Unsupported agent label template conversion '!{conversion}'")
        parts.append(formatter.format_field(value, format_spec))
    return ''.join(parts)


def render_agent_labels(cfg: Dict[str, Any]) -> None:
    agents = cfg.get("agents", [])
    if not isinstance(agents, list):
        return
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        label = agent.get("label")
        if isinstance(label, str) and "{" in label and "}" in label:
            agent["label"] = _render_agent_label_template(label, cfg)


def generate_configs(spec_path: str, output_dir: str) -> None:
    spec = _read_yaml(spec_path)
    sweep_name = spec.get('sweep_name')
    if not sweep_name:
        raise ValueError("sweep_name is required in sweep spec")
    raw_base_config = spec.get('base_config')
    if not raw_base_config:
        raise ValueError("base_config is required in sweep spec")
    spec_dir = Path(spec_path).parent
    # Strategy:
    #  1) If path is absolute and exists -> use it.
    #  2) If relative and exists as given (project-root relative) -> use it.
    #  3) Else try spec_dir / path.
    #  4) If still missing, raise with helpful diagnostics.
    candidate_paths = []
    p = Path(raw_base_config)
    if p.is_absolute():
        candidate_paths.append(p)
    else:
        candidate_paths.append(p)  # as provided (likely project-root relative)
        candidate_paths.append(spec_dir / p)  # relative to spec location
    chosen: Path | None = None
    for cand in candidate_paths:
        if cand.exists():
            chosen = cand.resolve()
            break
    if chosen is None:
        tried = "\n  - ".join(str(c.resolve()) for c in candidate_paths)
        raise FileNotFoundError(
            "base_config not found. Tried:\n  - " + tried + f"\n(spec_dir={spec_dir.resolve()})"
        )
    base_cfg = _read_yaml(str(chosen))

    overrides: Dict[str, Any] = spec.get('override', {}) or {}
    grid: Dict[str, List[Any]] = spec.get('grid', {}) or {}
    restrict: Dict[str, Any] = spec.get('restrict', {}) or {}
    combos = expand_grid(grid)
    combos = _restrict_combos(
        combos,
        max_combos=restrict.get('max_combinations'),
    )

    # Prepare (now mandatory) output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply overrides once to a fresh copy per combo
    written = 0
    for idx, combo in enumerate(combos, start=1):
        cfg = deepcopy(base_cfg)
        apply_overrides(cfg, overrides)
        for k, v in combo.items():
            apply_overrides(cfg, {k: v})
        render_agent_labels(cfg)
        # Ensure each agent has consistent policy params section if present
        # (We don't mutate beyond provided dotted keys.)
        run_path = out_dir / f"run_{idx:04d}.yaml"
        _write_yaml(run_path, cfg)
        written += 1

    print(f"[make_config_grid] Wrote {written} configs to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate a concrete experiment-config grid from a sweep spec")
    ap.add_argument('--spec', required=True, help='Path to sweep spec YAML')
    ap.add_argument('--output-dir', required=True, help='Directory to write generated run_<n>.yaml configs')
    args = ap.parse_args()
    generate_configs(args.spec, args.output_dir)


if __name__ == '__main__':
    main()
