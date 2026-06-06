#!/usr/bin/env python3
"""Run a directory of generated experiment configs against an experiment module.

Usage examples:
  # Sequential
  python -m pipeline.run_experiment_sweep --configs-dir sweeps/reinforce_unified_demo/configs \
      --experiment-module derby.experiments.one_camp_n_days \
      --output-dir results/reinforce_unified_demo

  # Parallel (4 workers)
  python -m pipeline.run_experiment_sweep --configs-dir sweeps/reinforce_unified_demo/configs \
      --experiment-module derby.experiments.one_camp_n_days \
      --output-dir results/reinforce_unified_demo --parallel 4

  # Dry run (print commands only)
  python -m pipeline.run_experiment_sweep --configs-dir sweeps/reinforce_unified_demo/configs \
      --experiment-module derby.experiments.one_camp_n_days \
      --output-dir results/reinforce_unified_demo --dry-run

Skips a run if a completion JSON already exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

COMPLETION_RECORD_NAME = "_RUN_COMPLETE.json"
FAILURE_RECORD_NAME = "failure.json"
MANIFEST_NAME = "run_summary.json"
MAX_STDERR_TAIL = 5000  # max chars of stderr tail we retain for failures

Result = Dict[str, Any]
PREFIX = "[run_experiment_sweep]"


def _discover_configs(cfg_dir: Path) -> List[Path]:
    """Return sorted list of run config YAML files in a directory."""
    return sorted([p for p in cfg_dir.glob('*.yaml') if p.is_file()])


def _build_command(cfg_path: Path, output_dir: Path, experiment_module: str) -> List[str]:
    """Build the subprocess command to execute a single experiment config."""
    return [
        sys.executable,
        '-m',
        experiment_module,
        '--config',
        str(cfg_path),
        '--output-dir',
        str(output_dir),
    ]


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def _write_completion_record(
    completion_record: Path,
    cfg_path: Path,
    run_dir: Path,
    experiment_module: str,
    start_ts: float,
    end_ts: float,
) -> None:
    payload = {
        "status": "ok",
        "config": str(cfg_path),
        "output_dir": str(run_dir.resolve()),
        "experiment_module": experiment_module,
        "start_time": _utc_iso(start_ts),
        "end_time": _utc_iso(end_ts),
        "duration_s": round(end_ts - start_ts, 4),
    }
    completion_record.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def _dirty_run_dirs(configs: List[Path], output_dir: Path) -> List[Path]:
    dirty: List[Path] = []
    for cfg_path in configs:
        run_dir = output_dir / cfg_path.stem
        if not run_dir.exists():
            continue
        if (run_dir / COMPLETION_RECORD_NAME).exists():
            continue
        if _is_empty_dir(run_dir):
            continue
        dirty.append(run_dir)
    return dirty


def _run_one(
    cfg_path: Path,
    output_dir: Path,
    experiment_module: str,
    dry_run: bool,
    index: int,
    total: int,
    status_interval: float,
) -> Result:
    """Execute (or simulate) a single config.

    Returns a result dict with keys: config, status, and optionally cmd / rc / stderr / error.
    """
    run_dir = output_dir / cfg_path.stem  # isolate each run's artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_record = run_dir / COMPLETION_RECORD_NAME
    if completion_record.exists():
        print(f"{PREFIX} SKIP  {index}/{total} {cfg_path.name} completion record exists", flush=True)
        return {
            "config": str(cfg_path),
            "status": "skipped",
            "output_dir": str(run_dir.resolve()),
            "experiment_module": experiment_module,
        }
    cmd = _build_command(cfg_path, run_dir, experiment_module)
    print(f"{PREFIX} START {index}/{total} {cfg_path.name} -> {run_dir}", flush=True)
    if dry_run:
        print(f"{PREFIX} DONE  {index}/{total} {cfg_path.name} dry-run -> {' '.join(cmd)}", flush=True)
        return {
            "config": str(cfg_path),
            "status": "dry-run",
            "cmd": ' '.join(cmd),
            "duration_s": 0.0,
            "output_dir": str(run_dir.resolve()),
            "experiment_module": experiment_module,
        }
    try:
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        next_status_ts = t0 + status_interval if status_interval > 0 else None
        while True:
            try:
                _stdout, stderr = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if next_status_ts is not None and time.time() >= next_status_ts:
                elapsed = time.time() - t0
                print(
                    f"{PREFIX} STATUS elapsed={elapsed:.1f}s done={index - 1}/{total} "
                    f"running=1 pending={total - index}",
                    flush=True,
                )
                next_status_ts += status_interval
        end_ts = time.time()
        dt = end_ts - t0
        if proc.returncode == 0:
            _write_completion_record(completion_record, cfg_path, run_dir, experiment_module, t0, end_ts)
            print(f"{PREFIX} DONE  {index}/{total} {cfg_path.name} ok duration={dt:.1f}s", flush=True)
            return {
                "config": str(cfg_path),
                "status": "ok",
                "duration_s": round(dt, 4),
                "output_dir": str(run_dir.resolve()),
                "experiment_module": experiment_module,
            }
        else:
            print(
                f"{PREFIX} DONE  {index}/{total} {cfg_path.name} failed rc={proc.returncode} "
                f"duration={dt:.1f}s",
                flush=True,
            )
            return {
                "config": str(cfg_path),
                "status": "failed",
                "rc": proc.returncode,
                "stderr": stderr[-MAX_STDERR_TAIL:],
                "duration_s": round(dt, 4),
                "output_dir": str(run_dir.resolve()),
                "experiment_module": experiment_module,
            }
    except Exception as e:  # pragma: no cover
        print(f"{PREFIX} DONE  {index}/{total} {cfg_path.name} error error={e}", flush=True)
        return {
            "config": str(cfg_path),
            "status": "error",
            "error": str(e),
            "duration_s": 0.0,
            "output_dir": str(run_dir.resolve()),
            "experiment_module": experiment_module,
        }


def _status_counts(results: List[Result]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    return counts


def _print_parallel_status(
    start_ts: float,
    results: List[Result],
    total: int,
    max_workers: int,
) -> None:
    done = len(results)
    active = min(max_workers, max(0, total - done))
    pending = max(0, total - done - active)
    counts = _status_counts(results)
    print(
        f"{PREFIX} STATUS elapsed={time.time() - start_ts:.1f}s done={done}/{total} "
        f"running={active} pending={pending} ok={counts.get('ok', 0)} "
        f"failed={counts.get('failed', 0)} error={counts.get('error', 0)} "
        f"skipped={counts.get('skipped', 0)} dry-run={counts.get('dry-run', 0)}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Run a generated experiment-config sweep")
    ap.add_argument('--configs-dir', required=True, help='Directory containing run_*.yaml configs')
    ap.add_argument('--experiment-module', required=True, help='Python module to invoke via `python -m` for each config')
    ap.add_argument('--output-dir', required=True, help='Directory root for per-run outputs')
    ap.add_argument('--parallel', type=int, default=1, help='Number of parallel workers (default 1)')
    ap.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    ap.add_argument(
        '--status-interval',
        type=float,
        default=10.0,
        help='Seconds between progress status lines while runs are active; set 0 to disable',
    )
    ap.add_argument(
        '--json',
        dest='json_out',
        action='store_true',
        help='Also print final run_summary.json payload to stdout for scripts/CI',
    )
    args = ap.parse_args()

    cfg_dir = Path(args.configs_dir)
    if not cfg_dir.is_dir():
        raise SystemExit(f"Config directory not found: {cfg_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _discover_configs(cfg_dir)
    if not configs:
        print("No configs discovered", file=sys.stderr)
        return
    print(f"{PREFIX} Discovered {len(configs)} configs in {cfg_dir}", flush=True)

    dirty = _dirty_run_dirs(configs, output_dir)
    if dirty:
        print(
            f"{PREFIX} ERROR: found non-empty run directorie(s) without {COMPLETION_RECORD_NAME}.",
            file=sys.stderr,
        )
        print(
            f"{PREFIX} ERROR: inspect/delete/move these directories or choose a new --output-dir:",
            file=sys.stderr,
        )
        for path in dirty:
            print(f"{PREFIX} ERROR:   {path}", file=sys.stderr)
        raise SystemExit(1)

    overall_start_ts = time.time()
    overall_start_iso = _utc_iso(overall_start_ts)
    results: List[Result] = []
    total_configs = len(configs)
    if args.parallel <= 1:
        for idx, p in enumerate(configs, start=1):
            r = _run_one(
                p,
                output_dir,
                args.experiment_module,
                args.dry_run,
                idx,
                total_configs,
                args.status_interval,
            )
            results.append(r)
    else:
        if args.parallel < 1:
            raise SystemExit("--parallel must be >= 1")
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            fut_map = {
                ex.submit(
                    _run_one,
                    p,
                    output_dir,
                    args.experiment_module,
                    args.dry_run,
                    idx,
                    total_configs,
                    0,
                ): p
                for idx, p in enumerate(configs, start=1)
            }
            pending_futs = set(fut_map)
            next_status_ts = overall_start_ts + args.status_interval if args.status_interval > 0 else None
            while pending_futs:
                done_futs, pending_futs = cf.wait(
                    pending_futs,
                    timeout=0.5,
                    return_when=cf.FIRST_COMPLETED,
                )
                for fut in done_futs:
                    results.append(fut.result())
                if next_status_ts is not None and time.time() >= next_status_ts and pending_futs:
                    _print_parallel_status(overall_start_ts, results, total_configs, args.parallel)
                    next_status_ts += args.status_interval

    # Summary
    counts = _status_counts(results)
    # Order summary for readability
    ordered_keys = [k for k in ['ok', 'skipped', 'failed', 'error', 'dry-run'] if k in counts] + [k for k in counts if k not in {'ok','skipped','failed','error','dry-run'}]
    ordered_summary = {k: counts[k] for k in ordered_keys}
    print(f"{PREFIX} Summary: {ordered_summary}", flush=True)

    overall_end_ts = time.time()
    overall_end_iso = _utc_iso(overall_end_ts)
    total_duration_s = round(overall_end_ts - overall_start_ts, 4)

    # Persist a manifest so failures/success can be inspected later without re-running.
    manifest_path = output_dir / MANIFEST_NAME
    manifest_payload = {
        "results": results,
        "summary": ordered_summary,
        "root_output_dir": str(output_dir.resolve()),
        "configs_dir": str(cfg_dir.resolve()),
        "experiment_module": args.experiment_module,
        "status_interval_s": args.status_interval,
        "start_time": overall_start_iso,
        "end_time": overall_end_iso,
        "total_duration_s": total_duration_s,
    }
    all_skipped = bool(results) and all(r.get('status') == 'skipped' for r in results)
    try:
        if all_skipped:
            print(f"{PREFIX} No manifest written because all configs were skipped", flush=True)
        else:
            with manifest_path.open('w', encoding='utf-8') as f:
                json.dump(manifest_payload, f, indent=2)
            print(f"{PREFIX} Wrote manifest {manifest_path}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"{PREFIX} WARNING: could not write manifest ({e})", file=sys.stderr)

    # Write failure detail files for post-mortem (stderr tail)
    for r in results:
        if r['status'] in {'failed', 'error'}:
            run_dir = output_dir / Path(r['config']).stem
            fail_path = run_dir / FAILURE_RECORD_NAME
            detail = {k: v for k, v in r.items() if k not in {'config'}}
            try:
                with fail_path.open('w', encoding='utf-8') as f:
                    json.dump(detail, f, indent=2)
            except Exception:
                pass

    if args.json_out:
        print(json.dumps(manifest_payload, indent=2))


if __name__ == '__main__':
    main()
