#!/usr/bin/env python3
"""Generate concrete experiment configs from a sweep spec (minimal version).

We intentionally keep labeling dead simple: the generated config's label is just the
policy class name string (e.g. "REINFORCE"). Any richer run identification should be
handled downstream via the policy instance __repr__ in logs or by explicit user fields.

Minimal spec keys:
    sweep_name (str, required)
    base_config (path, required)
    override (mapping, optional) - dotted-key overrides
    grid (mapping, optional) - dotted-key -> list for cartesian product
    restrict.max_combinations / restrict.sample_fraction (optional)

Example:
    sweep_name: demo
    base_config: configs/one_camp_n_days_v2_config.yaml
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
import os
import random
import sys
import inspect
import string
from derby.policies.reinforce import REINFORCE  # direct import keeps file simple
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


POLICY_REGISTRY: Dict[str, Any] = {
    'reinforce': REINFORCE,
}


def _resolve_policy_class(policy_token: Any):
    """Resolve a policy token to a known class via explicit registry.

    If token is already a class/object, return it unchanged. Returns None if unknown.
    """
    if not isinstance(policy_token, str):
        return policy_token
    return POLICY_REGISTRY.get(policy_token.lower())


def build_label(combo: Dict[str, Any], template: str | None, policy_cls: Any, learner_params: Dict[str, Any]) -> str:
    """Compose final label with strict template validation.

    Rules:
      - If no template: label == <PolicyClassName> (or 'policy' if unresolved).
      - If template provided: every placeholder must be a constructor (__init__) parameter
        name of the resolved policy class. If policy cannot be resolved, raise.
      - Context values are taken from: policy __init__ defaults, then learner_params, then
        sweep combo overrides (last wins). This lets templates reference params even when
        they are not part of the sweep (using their default or base-config value).
    """
    if policy_cls is None:
        if template:
            raise ValueError("Cannot apply label_template: policy class could not be resolved.")
        return "policy"

    policy_name = getattr(policy_cls, '__name__', str(policy_cls))
    if not template:
        # Attempt repr(policy_instance) for richer label if constructor allows a lightweight instantiation.
        try:
            sig = inspect.signature(policy_cls.__init__)
            # Build minimal kwargs: use defaults where available; required params get simple stand-ins.
            kwargs = {}
            for pname, p in sig.parameters.items():
                if pname == 'self':
                    continue
                if p.default is not inspect._empty:
                    kwargs[pname] = p.default
                else:
                    # Provide minimal stand-ins for known required params.
                    if pname == 'auction_item_spec_ids':
                        kwargs[pname] = [0]
                    elif pname == 'num_dist_per_spec':
                        kwargs[pname] = 1
                    else:
                        # Skip unknown required param; will likely raise
                        raise RuntimeError(f"Cannot auto-instantiate policy for repr; required param '{pname}' has no default")
            instance = policy_cls(**kwargs)
            return repr(instance)
        except Exception:
            # Fallback to simple class name if instantiation fails
            return policy_name

    # Introspect constructor parameters fresh each call (simplicity over micro-optimizing)
    sig = inspect.signature(policy_cls.__init__)
    param_defaults: Dict[str, Any] = {}
    valid_params: set[str] = set()
    for pname, p in sig.parameters.items():
        if pname == 'self':
            continue
        valid_params.add(pname)
        if p.default is not inspect._empty:
            param_defaults[pname] = p.default

    # Extract placeholder field names from template
    formatter = string.Formatter()
    field_names: set[str] = set()
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is not None and field_name != '':
            field_names.add(field_name)

    # Validate placeholders strictly
    invalid = [f for f in field_names if f not in valid_params]
    if invalid:
        raise ValueError(
            "label_template references non-policy parameters: "
            + ", ".join(invalid)
            + f". Valid policy params: {sorted(valid_params)}"
        )

    # Build context: defaults -> learner params -> combo short-key overrides
    ctx: Dict[str, Any] = dict(param_defaults)
    ctx.update(learner_params or {})
    for full_key, value in combo.items():
        short = full_key.split('.')[-1]
        if short in valid_params:  # only include if actually a policy param
            ctx[short] = value

    # Ensure all referenced placeholders have values (should, due to defaults or overrides)
    missing_values = [f for f in field_names if f not in ctx]
    if missing_values:
        raise ValueError(
            "Missing values for template fields: " + ", ".join(missing_values)
        )

    try:
        rendered = template.format(**ctx)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"label_template formatting failed; missing key '{missing}'. Context keys: {list(ctx)}"
        ) from e
    return f"{policy_name}{rendered}"


def generate_configs(spec_path: str, configs_dir: str) -> None:
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
    label_template: str | None = spec.get('label_template')
    seed = spec.get('seed')

    combos = expand_grid(grid)
    combos = _sample_combos(
        combos,
        max_combos=restrict.get('max_combinations'),
        sample_fraction=restrict.get('sample_fraction'),
        seed=seed,
    )

    # Prepare (now mandatory) output directory
    out_dir = Path(configs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply overrides once to a fresh copy per combo
    written = 0
    for idx, combo in enumerate(combos, start=1):
        cfg = deepcopy(base_cfg)
        apply_overrides(cfg, overrides)
        for k, v in combo.items():
            apply_overrides(cfg, {k: v})
        learner = cfg.get('agents', [{}])[0]
        learner_policy_token = learner.get('policy')
        learner_params = learner.get('params', {}) or {}
        policy_cls = _resolve_policy_class(learner_policy_token)
        cfg['label'] = build_label(combo, label_template, policy_cls, learner_params)
        # Ensure each agent has consistent policy params section if present
        # (We don't mutate beyond provided dotted keys.)
        run_path = out_dir / f"run_{idx:04d}.yaml"
        _write_yaml(run_path, cfg)
        written += 1

    print(f"[make_grid] Wrote {written} configs to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Generate concrete configs from a sweep spec")
    ap.add_argument('--spec', required=True, help='Path to sweep spec YAML')
    ap.add_argument('--configs-dir', required=True, help='Directory to write generated run_<n>.yaml configs')
    args = ap.parse_args()
    generate_configs(args.spec, args.configs_dir)


if __name__ == '__main__':
    main()
