#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import itertools
import logging
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Global tuning constant: fraction applied to theoretical (usable_mem / per_worker_mem)
# to provide headroom for allocator overhead, variance, transient peaks.
DEFAULT_MEMORY_SAFETY_FACTOR = 0.85
# Global memory reserve (MB) subtracted from MemTotal before capacity math; provides
# buffer for OS, parent process, allocator fragmentation, page cache, etc.
DEFAULT_MEMORY_RESERVE_MB = 1024

_SUPPRESS_TOKENS = {"NONE", "OFF", "QUIET", "SILENT", "NA", "N/A"}

def _normalize_log_level(val: str | None, default: str = "INFO") -> str | None:
	if val is None:
		return default
	v = val.strip().upper()
	if v in _SUPPRESS_TOKENS:
		return None  # sentinel meaning disable
	return v

def _configure_sweep_logging(level_str: str | None) -> None:
	# Configure only once; further calls are no-ops if handlers exist
	if logging.getLogger().handlers:
		# Still allow disabling if requested
		if level_str is None:
			logging.disable(logging.CRITICAL)
		else:
			try:
				logging.getLogger().setLevel(getattr(logging, level_str, logging.INFO))
			except Exception:
				logging.getLogger().setLevel(logging.INFO)
		return
	if level_str is None:
		logging.basicConfig(level=logging.CRITICAL)  # set then immediately disable
		logging.disable(logging.CRITICAL)
		return
	lvl = getattr(logging, level_str, logging.INFO)
	logging.basicConfig(
		level=lvl,
		format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
		datefmt="%H:%M:%S",
	)


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


def _worker_run(i: int, cfg: Dict[str, Any], parquet_dir: str,
			 tf_intra: int, tf_inter: int, sweep_log_level: str | None) -> Dict[str, Any]:
	# Ensure logging configured in spawned worker (spawn does not inherit handlers)
	try:
		_configure_sweep_logging(sweep_log_level)
	except Exception:
		# Fallback minimal config
		if not logging.getLogger().handlers:
			logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
	os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
	if tf_intra > 0:
		os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_intra)
	if tf_inter > 0:
		os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_inter)
	if tf_intra > 0:
		os.environ.setdefault("OMP_NUM_THREADS", str(tf_intra))
		os.environ.setdefault("MKL_NUM_THREADS", str(tf_intra))

	# --- Instrumentation: early memory + environment snapshot BEFORE TF import ---
	def _mem_mb() -> float | None:
		try:
			import psutil, os as _os  # type: ignore
			return psutil.Process(_os.getpid()).memory_info().rss / (1024 ** 2)
		except Exception:
			return None

	_mem0 = _mem_mb()
	if _mem0 is not None:
		logger.debug("[run %s] rss_before_tf_import=%.1fMB", i, _mem0)
	else:
		logger.debug("[run %s] rss_before_tf_import=unknown (psutil not available)", i)

	# Attempt a controlled TensorFlow import so we can log version & memory deltas.
	_tf_import_err: str | None = None
	try:
		import tensorflow as _tf  # noqa: F401
		logger.debug("[run %s] tensorflow_version=%s", i, getattr(_tf, '__version__', 'unknown'))
		_mem1 = _mem_mb()
		if _mem1 is not None and _mem0 is not None:
			logger.debug("[run %s] rss_after_tf_import=%.1fMB (+%.1fMB)", i, _mem1, _mem1 - _mem0)
	except Exception as e:  # pragma: no cover (only for diagnostic path)
		_tf_import_err = f"{type(e).__name__}: {e}"
		logger.error("[run %s] tensorflow_import_failed=%s", i, _tf_import_err)

	import uuid
	from derby.experiments.one_camp_n_days_v2 import run_experiment_from_config

	# cfg already contains its final label assigned in the parent process
	cfg = deepcopy(cfg)
	# Do NOT set or override any seed here; if the user supplied one in cfg, leave it.
	cfg["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

	# Pre-generate run_id so we can advertise it even if early failure occurs.
	run_id: str | None = str(uuid.uuid4())
	try:
		policy0 = cfg.get("agents", [{}])[0].get("policy")
		lr0 = cfg.get("agents", [{}])[0].get("params", {}).get("learning_rate")
		ne = cfg.get("num_epochs")
		nt = cfg.get("num_trajs")
		logger.info("[run %s] starting run_id=%s label=%s policy=%s lr=%s epochs=%s trajs=%s", i, run_id, cfg['label'], policy0, lr0, ne, nt)
	except Exception:
		logger.info("[run %s] starting run_id=%s label=%s", i, run_id, cfg['label'])

	# Simple background sampler to record peak RSS during the experiment run.
	import threading as _threading
	_peak_holder = {"peak": _mem0}
	_stop_evt = _threading.Event()

	def _sample_loop():  # lightweight polling (sleep dominates)
		while not _stop_evt.is_set():
			m = _mem_mb()
			if m is not None:
				cur_peak = _peak_holder.get("peak")
				if cur_peak is None or m > cur_peak:
					_peak_holder["peak"] = m
			_stop_evt.wait(0.5)  # 500ms cadence keeps overhead negligible

	_sampler_thread = _threading.Thread(target=_sample_loop, name=f"memsampler-{i}", daemon=True)

	start = time.time()
	status = "ok"
	error = None
	try:
		# Suppress experiment module logs only (leave instrumentation + start lines). If
		# debugging is desired, set env DERBY_DEBUG_INNER=1 to keep logs.
		import logging as _logging  # local import to avoid affecting parent before fork/spawn
		if os.environ.get("DERBY_DEBUG_INNER", "0").upper() not in {"1", "TRUE", "YES", "ON"}:
			_logging.disable(_logging.CRITICAL)
		# Start peak memory sampler just before the heavy lifting begins.
		try:
			_sampler_thread.start()
		except Exception:
			pass
		# Pass a pre-generated run_id so even early failures have a known identifier.
		run_id = run_experiment_from_config(
			cfg,
			output_dir_override=parquet_dir,
			run_id=run_id,
		)
	except Exception as e:
		status = "failed"
		error = f"{type(e).__name__}: {e}"
	dur = time.time() - start


	# Post-run memory snapshot (only if experiment got that far)
	_mem2 = _mem_mb()
	if _mem2 is not None:
		logger.info("[run %s] rss_post_run=%.1fMB", i, _mem2)

	# Prepare memory stats (may be None if psutil missing or snapshot failed)
	def _fmt(x):
		return float(x) if isinstance(x, (int, float)) else None
	mem_start = _fmt(_mem0)
	mem_end = _fmt(_mem2)
	# Stop sampler and capture peak
	try:
		_stop_evt.set()
		if _sampler_thread.is_alive():
			_sampler_thread.join(timeout=1.0)
	except Exception:
		pass
	mem_delta = _fmt((mem_end - mem_start) if (mem_end is not None and mem_start is not None) else None)
	mem_peak = _fmt(_peak_holder.get("peak"))
	return {
		"index": i,
		"status": status,
		"error": error,
		"duration_s": dur,
		"run_id": run_id,
		"rss_mb_start": mem_start,
		"rss_mb_end": mem_end,
		"rss_mb_delta": mem_delta,
		"rss_mb_peak": mem_peak,
	}


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
	# Unified output management
	ap.add_argument(
		"--output-dir",
		type=str,
		default="results",
		help="Base output directory. Parquet files go to <output-dir>/parquet; results JSONL is written in <output-dir>."
	)
	# Label handling: we no longer accept a label prefix. Each variant's label is the existing
	# (possibly user-specified) label with a deterministic suffix -i<index>. If the base config
	# lacks a label entirely, we derive one from the grid YAML stem. This keeps semantics simple
	# and avoids extra CLI knobs.
	# Removed --base-seed: this orchestrator no longer manages RNG seeding; supply seeds in YAML if desired.
	ap.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2)), help="Parallel worker processes")
	ap.add_argument("--tf-intra", type=int, default=1, help="TF intra-op threads per process (>=1)")
	ap.add_argument("--tf-inter", type=int, default=1, help="TF inter-op threads per process (>=1)")
	ap.add_argument(
		"--start-method",
		choices=["spawn", "fork", "forkserver"],
		default="spawn",
		help="Multiprocessing start method. 'spawn' is safest with TensorFlow (avoids forking a process with complex native threads)."
	)
	ap.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		help="Log level for sweep orchestrator (DEBUG, INFO, WARNING, ERROR, NONE). Inner experiments use their own defaults. Suppression tokens: NONE/OFF/QUIET/SILENT/NA/N/A",
	)
	args = ap.parse_args(argv)

	# Configure sweep-level logging early
	_sweep_level = _normalize_log_level(args.log_level)
	_configure_sweep_logging(_sweep_level)

	# --- Per-run timestamped directory (avoid overwriting previous sweeps) ---
	base_out = Path(args.output_dir)
	base_out.mkdir(parents=True, exist_ok=True)
	timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
	run_dir = base_out / f"sweep_{timestamp}"
	parquet_path = run_dir / "parquet"
	parquet_path.mkdir(parents=True, exist_ok=True)
	run_dir.mkdir(parents=True, exist_ok=True)

	# Function to detect total memory (MB) for advisory only
	def _mem_total_mb() -> float | None:
		try:
			with open("/proc/meminfo", "r", encoding="utf-8") as f:
				for line in f:
					if line.startswith("MemTotal:"):
						parts = line.split()
						return float(parts[1]) / 1024.0
			return None
		except Exception:
			return None

	mem_total = _mem_total_mb()
	if mem_total:
		usable = max(0.0, mem_total - DEFAULT_MEMORY_RESERVE_MB)
		logger.info(
			"Memory advisory (pre-run): MemTotal=%.0fMB reserve=%sMB usable=%.0fMB safety_factor=%.2f",
			mem_total,
			DEFAULT_MEMORY_RESERVE_MB,
			usable,
			DEFAULT_MEMORY_SAFETY_FACTOR,
		)
		# write advisory file (initial, simplified)
		advisory_file = run_dir / "memory_advisory.txt"
		with open(advisory_file, "w", encoding="utf-8") as adv:
			adv.write("Pre-run Memory Advisory\n")
			adv.write(f"MemTotal_MB: {mem_total:.0f}\n")
			adv.write(f"Reserve_MB: {DEFAULT_MEMORY_RESERVE_MB}\n")
			adv.write(f"Usable_MB: {usable:.0f}\n")
			adv.write(f"SafetyFactor: {DEFAULT_MEMORY_SAFETY_FACTOR:.2f}\n")
	else:
		advisory_file = run_dir / "memory_advisory.txt"
		with open(advisory_file, "w", encoding="utf-8") as adv:
			adv.write("Could not determine MemTotal (missing /proc/meminfo).\n")

	if not args.base_yaml or not Path(args.base_yaml).exists():
		raise FileNotFoundError(f"--base-yaml file not found: {args.base_yaml}")
	base_cfg: Dict[str, Any] = _load_yaml_mapping(Path(args.base_yaml))

	if not args.grid_yaml or not Path(args.grid_yaml).exists():
		raise FileNotFoundError(f"--grid-yaml file not found: {args.grid_yaml}")
	loaded = _load_yaml_mapping(Path(args.grid_yaml))
	param_grid: Dict[str, List[Any]] = {str(k): v for k, v in loaded.items()}

	variants = expand_grid(param_grid)
	configs = [apply_overrides(base_cfg, ov) for ov in variants]

	# Determine a base label stem if none provided in configs
	grid_stem = Path(args.grid_yaml).stem
	for i, cfg in enumerate(configs):
		base_label = cfg.get("label") or base_cfg.get("label") or grid_stem or "run"
		cfg["label"] = f"{base_label}-i{i}"

	# Output locations now isolated per sweep run
	parquet_dir = str(parquet_path)
	results_jsonl_stable = run_dir / "parallel_results.jsonl"  # stable within run_dir
	results_jsonl_ts = run_dir / f"parallel_results_{timestamp}.jsonl"
	logger.info("Output directory for this sweep: %s", run_dir)

	logger.info("Running %s configs with %s workers; Parquet -> %s", len(configs), args.max_workers, parquet_dir)
	logger.info("TF threads per proc: intra=%s, inter=%s", args.tf_intra, args.tf_inter)

	results: List[Dict[str, Any]] = []
	overall_start = time.time()
	# Obtain a multiprocessing context; fallback gracefully if unavailable (older Python / platforms)
	try:
		ctx = mp.get_context(args.start_method)
	except (ValueError, AttributeError):  # platform may not support requested method
		logger.warning("Requested start method '%s' not available; falling back to default context", args.start_method)
		ctx = None

	executor_kwargs = {"max_workers": args.max_workers}
	if ctx is not None:
		executor_kwargs["mp_context"] = ctx

	with ProcessPoolExecutor(**executor_kwargs) as ex:
		futures = {}
		for i, cfg in enumerate(configs):
			fut = ex.submit(_worker_run, i, cfg, parquet_dir, args.tf_intra, args.tf_inter, _sweep_level)
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
			rid = res.get('run_id')
			if rid:
				logger.info("[run %s] finished run_id=%s status=%s elapsed=%s", i, rid, res['status'], dur_str)
			else:
				logger.info("[run %s] finished status=%s elapsed=%s", i, res['status'], dur_str)

	overall_end = time.time()
	wall_time = overall_end - overall_start
	total_cpu_time = sum((r.get('duration_s') or 0.0) for r in results)
	speedup = (total_cpu_time / wall_time) if wall_time > 0 else float('nan')
	efficiency = (speedup / args.max_workers) if args.max_workers > 0 else float('nan')

	# Write results to both stable and timestamped files
	for out_path in (results_jsonl_stable, results_jsonl_ts):
		with open(out_path, "w", encoding="utf-8") as f:
			for rec in sorted(results, key=lambda r: r["index"]):
				# Ensure run_id exists (may be None when a run fails before producing one)
				if "run_id" not in rec:
					rec = {**rec, "run_id": None}
				f.write(json.dumps(rec) + "\n")

	ok = sum(1 for r in results if r.get("status") == "ok")
	fail = len(results) - ok

	# Diagnostic: detect parquet files whose run_ids are not in our results list (possible leftovers or stray writes)
	try:
		if os.path.isdir(parquet_dir):
			result_run_ids = {r.get("run_id") for r in results if r.get("run_id")}
			parquet_files = [f for f in os.listdir(parquet_dir) if f.startswith("epoch_agg__") and f.endswith('.parquet')]
			parquet_run_ids = set()
			for fname in parquet_files:
				# epoch_agg__<uuid>.parquet
				base = fname[len("epoch_agg__"):-len('.parquet')]
				parquet_run_ids.add(base)
			unexpected = sorted(parquet_run_ids - result_run_ids)
			missing = sorted(result_run_ids - parquet_run_ids)
			if unexpected:
				logger.warning("Parquet files present with unknown run_ids (not in results JSON): %s", unexpected)
			if missing:
				logger.warning("Runs marked ok but missing parquet files: %s", missing)
	except Exception as _diag_e:  # pragma: no cover
		logger.debug("Parquet diagnostic skipped due to error: %s", _diag_e)
	logger.info(
		"Done: %s ok, %s failed. Results -> %s\nOverall wall time: %.2fs | Aggregate CPU time: %.2fs | Speedup: %.2fx | Efficiency: %.2f%% (workers=%s)",
	ok,
	fail,
	str(results_jsonl_ts),
	wall_time,
	total_cpu_time,
	speedup,
	(efficiency * 100.0),
	args.max_workers,
	)

	# Memory summary (skip None values)
	def _clean(values):
		vals = [v for v in values if isinstance(v, (int, float))]
		return vals
	mb_start = _clean([r.get("rss_mb_start") for r in results])
	mb_end = _clean([r.get("rss_mb_end") for r in results])
	mb_delta = _clean([r.get("rss_mb_delta") for r in results])
	mb_peak = _clean([r.get("rss_mb_peak") for r in results])

	def _stats(vals):
		if not vals:
			return "n/a"
		import statistics as _stats
		return f"mean={_stats.fmean(vals):.1f}MB max={max(vals):.1f}MB"

	logger.info(
		"Memory summary: start[%s] end[%s] delta[%s] peak[%s] (runs=%s)",
		_stats(mb_start),
		_stats(mb_end),
		_stats(mb_delta),
		_stats(mb_peak),
		len(results),
	)
	# Append post-run observed advisory
	try:
		if mem_total and (mb_end or mb_peak):
			import statistics as _stats
			usable = max(0.0, mem_total - DEFAULT_MEMORY_RESERVE_MB)
			def _percentile(vals, p):
				if not vals:
					return None
				vals_sorted = sorted(vals)
				k = max(0, min(len(vals_sorted) - 1, int(round((p / 100.0) * (len(vals_sorted) - 1)))))
				return float(vals_sorted[k])
			mean_end = _stats.fmean(mb_end) if mb_end else None
			median_end = _stats.median(mb_end) if mb_end else None
			p95_end = _percentile(mb_end, 95) if mb_end else None
			max_end = max(mb_end) if mb_end else None
			mean_peak = _stats.fmean(mb_peak) if mb_peak else None
			p95_peak = _percentile(mb_peak, 95) if mb_peak else None
			max_peak = max(mb_peak) if mb_peak else None
			# Aggressive uses mean end, conservative uses p95 peak if available else p95 end else mean end.
			agg_basis = mean_end or 0.0
			cons_basis = p95_peak or p95_end or mean_end or 0.0
			agg_recommended = int((usable / agg_basis) * DEFAULT_MEMORY_SAFETY_FACTOR) if agg_basis > 0 else 0
			cons_recommended = int((usable / cons_basis) * DEFAULT_MEMORY_SAFETY_FACTOR) if cons_basis > 0 else 0
			logger.info(
				"Memory advisory (observed): mean_end=%.1fMB p95_end=%sMB mean_peak=%sMB p95_peak=%sMB usable=%.0fMB safety_factor=%.2f -> max_workers aggressive=%s conservative=%s",
				(mean_end if mean_end is not None else float('nan')),
				f"{p95_end:.1f}" if p95_end is not None else "n/a",
				f"{mean_peak:.1f}" if mean_peak is not None else "n/a",
				f"{p95_peak:.1f}" if p95_peak is not None else "n/a",
				usable,
				DEFAULT_MEMORY_SAFETY_FACTOR,
				agg_recommended,
				cons_recommended,
			)
			with open(advisory_file, "a", encoding="utf-8") as adv:
				adv.write("\nPost-run Observed Advisory\n")
				if mean_end is not None: adv.write(f"MeanRSS_End_MB: {mean_end:.1f}\n")
				if median_end is not None: adv.write(f"MedianRSS_End_MB: {median_end:.1f}\n")
				if p95_end is not None: adv.write(f"P95RSS_End_MB: {p95_end:.1f}\n")
				if max_end is not None: adv.write(f"MaxRSS_End_MB: {max_end:.1f}\n")
				if mean_peak is not None: adv.write(f"MeanPeakRSS_MB: {mean_peak:.1f}\n")
				if p95_peak is not None: adv.write(f"P95PeakRSS_MB: {p95_peak:.1f}\n")
				if max_peak is not None: adv.write(f"MaxPeakRSS_MB: {max_peak:.1f}\n")
				adv.write(f"SafetyFactor: {DEFAULT_MEMORY_SAFETY_FACTOR:.2f}\n")
				adv.write(f"RecommendedMaxWorkers_Aggressive: {agg_recommended}\n")
				adv.write(f"RecommendedMaxWorkers_Conservative: {cons_recommended}\n")
	except Exception:
		pass
	return 0 if fail == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
