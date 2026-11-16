"""Launch the full corrected Stage 1 matrix as parallel per-(model, power, seed) jobs.

Usage: py -3.13 run_stage1_batch.py --output-dir stage1_validated_outputs/full_10seed --workers 12

Fleets are created once before any job starts. Each job is resumable (run_stage1.py
skips completed cases), so re-running this command continues an interrupted batch.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import stage1_common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--models", nargs="+", default=["v2g", "v1g"])
    parser.add_argument("--powers", nargs="+", type=int, default=[11, 22])
    parser.add_argument("--day-type", default="weekday")
    args = parser.parse_args()
    out = Path(args.output_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        _, path, meta = c.ensure_fleet(c.fleet_config(seed, args.day_type))
        print(f"fleet seed {seed}: {path.name} {meta['fleet_hash'][:12]}", flush=True)

    jobs = [(m, p, s) for m in args.models for p in args.powers for s in args.seeds]

    def run(job):
        model, power, seed = job
        log = out / "logs" / f"{model}_{power}kW_seed{seed}.log"
        cmd = [sys.executable, "-W", "ignore", "run_stage1.py", "--model", model, "--power-kw", str(power),
               "--seeds", str(seed), "--day-type", args.day_type, "--output-dir", str(out)]
        started = time.time()
        with open(log, "a", encoding="utf-8") as handle:
            code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=c.ROOT)
        print(f"{time.strftime('%H:%M:%S')} done {model} {power}kW seed{seed} exit={code} "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
        return code

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(run, jobs))
    failed = [job for job, code in zip(jobs, codes) if code != 0]
    print("ALL DONE" if not failed else f"FAILED JOBS: {failed}", flush=True)


if __name__ == "__main__":
    main()
