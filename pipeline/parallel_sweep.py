#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
	parts = dotted_key.split('.')
	cur = cfg
	for i, k in enumerate(parts):
		is_last = (i == len(parts) - 1)
		if k.isdigit():
			idx = int(k)
			if not isinstance(cur, list):
				raise TypeError(f"Expected list at {'.'.join(parts[:i])}, got {type(cur).__name__}")
			if idx >= len(cur):
				raise IndexError(f"Index {idx} out of range for key path {dotted_key}")
			if is_last:
				cur[idx] = value
			else:
				cur = cur[idx]
		else:
			if is_last:
				cur[k] = value
			else:
				if k not in cur or not isinstance(cur[k], (dict, list)):
					cur[k] = {}
				cur = cur[k]


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
	if not grid:
		return [{}]
	keys = list(grid.keys())
	values_product = itertools.product(*(grid[k] for k in keys))
	return [{k: v for k, v in zip(keys, combo)} for combo in values_product]


def apply_overrides(base_cfg: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	cfg = deepcopy(base_cfg)
	for k, v in override.items():
		set_by_dotted_key(cfg, k, v)
	return cfg


def _worker_run(i: int, cfg: Dict[str, Any], parquet_dir: str, label_prefix: str,
				 base_seed: int, tf_intra: int, tf_inter: int) -> Dict[str, Any]:
	os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
	if tf_intra > 0:
		os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_intra)
	if tf_inter > 0:
		os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_inter)
	if tf_intra > 0:
		os.environ.setdefault("OMP_NUM_THREADS", str(tf_intra))
		os.environ.setdefault("MKL_NUM_THREADS", str(tf_intra))

	try:
		repo_root = Path(__file__).resolve().parent.parent
		if str(repo_root) not in sys.path:
			sys.path.insert(0, str(repo_root))
	except Exception:
		pass

	import random
	import numpy as np
	from derby.experiments.one_camp_n_days_v2 import _run_simple_config

	run_seed = (base_seed + i) % (2**31 - 1)
	cfg = deepcopy(cfg)
	cfg["label"] = f"{label_prefix}-i{i}"
	cfg["seed"] = run_seed
	cfg["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

	try:
		policy0 = cfg.get("agents", [{}])[0].get("policy")
		lr0 = cfg.get("agents", [{}])[0].get("params", {}).get("learning_rate")
		ne = cfg.get("num_epochs")
		nt = cfg.get("num_trajs")
		print(f"[run {i}] starting label={cfg['label']} policy={policy0} lr={lr0} epochs={ne} trajs={nt}")
	except Exception:
		print(f"[run {i}] starting label={cfg['label']}")

	random.seed(run_seed)
	np.random.seed(run_seed)

	start = time.time()
	status = "ok"
	error = None
	try:
		_run_simple_config(cfg, output_dir_override=parquet_dir)
	except Exception as e:
		status = "failed"
		error = f"{type(e).__name__}: {e}"
	dur = time.time() - start
	return {"index": i, "status": status, "error": error, "duration_s": dur}


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
	try:
		import yaml  # type: ignore
	except Exception as e:
		raise RuntimeError("PyYAML is required to load YAML files. Install with poetry install --with dev or add pyyaml.") from e
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise ValueError(f"YAML file {path} did not contain a mapping/dict at the top level.")
	return data


def main(argv: List[str] | None = None) -> int:
	ap = argparse.ArgumentParser(description="Parallel sweep runner for Derby experiments")
	ap.add_argument("--base-yaml", type=str, required=True, help="Path to base config YAML file (.yaml/.yml)")
	ap.add_argument("--grid-yaml", type=str, required=True, help="Path to param grid YAML file (.yaml/.yml)")
	ap.add_argument("--parquet-dir", type=str, default="results/parquet", help="Output dir for Parquet logs")
	ap.add_argument("--label-prefix", type=str, default="sweep-par", help="Prefix for per-run labels")
	ap.add_argument("--base-seed", type=int, default=1337, help="Base seed for deterministic per-run seeds")
	ap.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2)), help="Parallel worker processes")
	ap.add_argument("--tf-intra", type=int, default=1, help="TF intra-op threads per process (>=1)")
	ap.add_argument("--tf-inter", type=int, default=1, help="TF inter-op threads per process (>=1)")
	ap.add_argument("--results-jsonl", type=str, default="results/sweep/parallel_results.jsonl", help="Path to append JSONL run results")
	ap.add_argument("--debug", action="store_true", help="Enable verbose training logs in all runs (overrides YAML configs)")
	args = ap.parse_args(argv)

	if not args.base_yaml or not Path(args.base_yaml).exists():
		raise FileNotFoundError(f"--base-yaml file not found: {args.base_yaml}")
	base_cfg: Dict[str, Any] = _load_yaml_mapping(Path(args.base_yaml))

	if not args.grid_yaml or not Path(args.grid_yaml).exists():
		raise FileNotFoundError(f"--grid-yaml file not found: {args.grid_yaml}")
	loaded = _load_yaml_mapping(Path(args.grid_yaml))
	param_grid: Dict[str, List[Any]] = {str(k): v for k, v in loaded.items()}

	variants = expand_grid(param_grid)
	configs = [apply_overrides(base_cfg, ov) for ov in variants]

	if args.debug:
		for cfg in configs:
			cfg["debug"] = True

	parquet_dir = args.parquet_dir
	Path(parquet_dir).mkdir(parents=True, exist_ok=True)
	out_dir = Path(args.results_jsonl).parent
	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"Running {len(configs)} configs with {args.max_workers} workers; Parquet -> {parquet_dir}")
	print(f"TF threads per proc: intra={args.tf_intra}, inter={args.tf_inter}")

	results: List[Dict[str, Any]] = []
	overall_start = time.time()
	with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
		futures = {}
		for i, cfg in enumerate(configs):
			fut = ex.submit(_worker_run, i, cfg, parquet_dir, args.label_prefix, args.base_seed, args.tf_intra, args.tf_inter)
			futures[fut] = i
		for fut in as_completed(futures):
			i = futures[fut]
			try:
				res = fut.result()
			except Exception as e:
				res = {"index": i, "status": "failed", "error": f"{type(e).__name__}: {e}", "duration_s": None}
			results.append(res)
			dur = res.get('duration_s')
			dur_str = f"{dur:.2f}s" if isinstance(dur, (int, float)) and dur is not None else "n/a"
			print(f"[run {i}] status={res['status']} elapsed={dur_str}")

	overall_end = time.time()
	wall_time = overall_end - overall_start
	total_cpu_time = sum((r.get('duration_s') or 0.0) for r in results)
	speedup = (total_cpu_time / wall_time) if wall_time > 0 else float('nan')
	efficiency = (speedup / args.max_workers) if args.max_workers > 0 else float('nan')

	with open(args.results_jsonl, "w", encoding="utf-8") as f:
		for rec in sorted(results, key=lambda r: r["index"]):
			f.write(json.dumps(rec) + "\n")

	ok = sum(1 for r in results if r.get("status") == "ok")
	fail = len(results) - ok
	print(
		"Done: {ok} ok, {fail} failed. Results -> {out}\n"
		"Overall wall time: {wall:.2f}s | Aggregate CPU time: {cpu:.2f}s | "
		"Speedup: {spd:.2f}x | Efficiency: {eff:.2%} (workers={workers})".format(
			ok=ok,
			fail=fail,
			out=args.results_jsonl,
			wall=wall_time,
			cpu=total_cpu_time,
			spd=speedup,
			eff=efficiency,
			workers=args.max_workers,
		)
	)
	return 0 if fail == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
