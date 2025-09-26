# Memory Advisory & Concurrency Guidance

This document explains how the parallel sweep process (`pipeline/parallel_sweep.py`) derives its memory advisory output and how to use it to choose a safe level of parallelism.

## Overview
Each sweep run writes a file `memory_advisory.txt` inside the timestamped sweep directory:
```
results/<your-output>/sweep_<YYYYMMDD-HHMMSS>/memory_advisory.txt
```
It contains a pre-run section and a post-run section.

## Global Constants
Defined at the top of `pipeline/parallel_sweep.py`:
- `DEFAULT_MEMORY_SAFETY_FACTOR = 0.85`  
  Multiplier applied to the theoretical capacity `(usable_memory / per_worker_memory)` to leave headroom for allocator overhead, variance, and short unobserved spikes.
- `DEFAULT_MEMORY_RESERVE_MB = 1024`  
  Always subtracted from total system memory (`/proc/meminfo: MemTotal`) before any division. This protects space for the OS, parent process, page cache, fragmentation, and future growth.

These are **not** CLI flags (kept intentionally simple); adjust them in the source if you need different defaults for your environment.

## Pre-run Section
Example:
```
Pre-run Memory Advisory
MemTotal_MB: 11940
Reserve_MB: 1024
Usable_MB: 10916
SafetyFactor: 0.85
```
Fields:
- **MemTotal_MB**: Detected from `/proc/meminfo` (in MB). If unavailable, only a fallback message is written.
- **Reserve_MB**: The fixed `DEFAULT_MEMORY_RESERVE_MB` buffer.
- **Usable_MB**: `max(0, MemTotal_MB - Reserve_MB)`; the pool considered allocatable by workers.
- **SafetyFactor**: The applied global multiplier, shown for transparency.

No worker-specific numbers appear yet because the sweep has not run—this section is informational only.

## Post-run Section
Example (fields present depend on data availability):
```
Post-run Observed Advisory
MeanRSS_End_MB: 1406.7
MedianRSS_End_MB: 1402.1
P95RSS_End_MB: 1508.3
MaxRSS_End_MB: 1519.4
MeanPeakRSS_MB: 1429.6
P95PeakRSS_MB: 1532.2
MaxPeakRSS_MB: 1540.8
SafetyFactor: 0.85
RecommendedMaxWorkers_Aggressive: 6
RecommendedMaxWorkers_Conservative: 6
```
Metrics are computed over successful runs only (failed runs with no memory samples are skipped):
- **MeanRSS_End_MB / Median / P95 / Max**: Resident Set Size after the experiment finishes (steady-state footprint).
- **MeanPeakRSS_MB / P95Peak / MaxPeak**: Highest sampled RSS during the run (500 ms sampling cadence). Captures transient spikes (e.g. TensorFlow graph build) that may be higher than end RSS.

## Recommendations
Two capacity suggestions are produced:
- **Aggressive**: `floor((Usable_MB / MeanRSS_End_MB) * SafetyFactor)`  
  Optimistic; suitable when peak ≈ end and variance is low.
- **Conservative**: Uses `p95_peak` if available, else `p95_end`, else falls back to `mean_end`:  
  `floor((Usable_MB / basis) * SafetyFactor)`  
  Safer when peak >> end or variance is noticeable.

If the aggressive and conservative values differ, prefer the conservative one unless turnaround time strongly outweighs risk of an occasional memory exhaustion.

## Why we keep a Safety Factor
Even percentile and peak sampling underestimates can occur due to:
1. Sub-sampling (spikes between 500 ms intervals)
2. Allocator fragmentation and hidden arenas
3. Additional parent / OS / cache usage
4. Simultaneous phase alignment (all workers hitting a heavy step concurrently)
5. Future code or model changes increasing per-worker footprint

A fixed 0.85 multiplier is a pragmatic default seen in many production memory sizing guidelines for dynamic ML workloads.

## When You Might Change Constants
Consider editing the constants if:
- Your environment is very stable and you want maximum throughput (raise factor toward 0.9–0.95, maybe shrink reserve).
- You see occasional OOM kills: decrease safety factor (e.g. 0.8) or raise reserve (e.g. 1536 MB).
- You add larger models or more libraries: temporarily lower factor until new memory behavior stabilizes.

## Interpreting Divergence
If `MaxPeakRSS_MB` is much larger (e.g. >1.2×) than `MeanRSS_End_MB`:
- Favor the conservative recommendation.
- Consider staggering worker startups (future enhancement) to reduce simultaneous peak overlap.

## FAQ
**Q: Why not auto-scale workers?**  
Transparency and reproducibility—implicit scaling hides true configured intent. Advisory-only keeps control explicit.

**Q: Why 500 ms sampling?**  
Balance of overhead (<1% typical) vs. granularity. Lower intervals add overhead; higher risk missing short spikes.

**Q: What if sampling misses a spike?**  
Safety factor + conservative percentile mitigate. If you still OOM, lower the factor or reserve.

**Q: Do failed runs skew stats?**  
Failed runs that never reached first sample are excluded from aggregates.

## Quick Checklist for Choosing Workers
1. Run a small sweep with representative configs.
2. Open `memory_advisory.txt`.
3. Look at `P95PeakRSS_MB` (or `P95RSS_End_MB` if peak missing).
4. Conservative recommended value is usually your safe baseline.
5. Increase by 1 and monitor if you want to probe headroom.

---
For further improvements (staggered starts, adaptive safety factor), contributions welcome.
