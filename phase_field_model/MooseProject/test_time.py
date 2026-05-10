#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path
from statistics import mean


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TIME_STEP_RE = re.compile(
    r"Time Step\s+(\d+),\s*time\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:,\s*dt\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))?"
)

FINISHED_RE = re.compile(
    r"Finished Executing\s*\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*s\s*\]"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(path: Path, target_sim_time: float = 1.0) -> dict:
    text = strip_ansi(path.read_text(errors="ignore"))

    steps = []
    for m in TIME_STEP_RE.finditer(text):
        step = int(m.group(1))
        sim_time = float(m.group(2))
        dt = float(m.group(3)) if m.group(3) is not None else None
        steps.append((step, sim_time, dt))

    finished_matches = FINISHED_RE.findall(text)
    total_wall_s = float(finished_matches[-1]) if finished_matches else None

    if not steps:
        return {
            "file": path.name,
            "ok": False,
            "reason": "No Time Step lines found",
        }

    final_step, final_sim_time, final_dt = max(steps, key=lambda x: x[0])

    avg_wall_per_step = None
    if total_wall_s is not None and final_step > 0:
        avg_wall_per_step = total_wall_s / final_step

    # 找到第一次达到或超过 target_sim_time 的 step
    reached_steps = [s for s in steps if s[1] >= target_sim_time]
    first_reach_step = min(reached_steps, key=lambda x: x[0])[0] if reached_steps else None

    # 估算方法 1：按 step 数比例估算 0-target_sim_time 的 wall time
    estimated_wall_to_target_by_step = None
    if avg_wall_per_step is not None and first_reach_step is not None:
        estimated_wall_to_target_by_step = avg_wall_per_step * first_reach_step

    # 估算方法 2：按模拟物理时间比例估算
    estimated_wall_to_target_by_sim_time = None
    if total_wall_s is not None and final_sim_time > 0 and final_sim_time >= target_sim_time:
        estimated_wall_to_target_by_sim_time = total_wall_s * target_sim_time / final_sim_time

    return {
        "file": path.name,
        "ok": True,
        "total_wall_s": total_wall_s,
        "final_step": final_step,
        "final_sim_time": final_sim_time,
        "final_dt": final_dt,
        "avg_wall_per_step_s": avg_wall_per_step,
        "first_step_reaching_target": first_reach_step,
        "estimated_wall_0_to_target_by_step_s": estimated_wall_to_target_by_step,
        "estimated_wall_0_to_target_by_sim_time_s": estimated_wall_to_target_by_sim_time,
    }


def fmt(x, ndigits=6):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{ndigits}g}"
    return str(x)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MOOSE .out logs for average wall time per step and estimated wall time from 0s to target simulation time."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./logs",
        help="Directory containing .out files. Default: current directory.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=1000.0,
        help="Target simulation time, default 1.0.",
    )
    parser.add_argument(
        "--pattern",
        default="*.out",
        help="File pattern, default *.out.",
    )
    parser.add_argument(
        "--csv",
        default="moose_time_summary.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob(args.pattern))

    if not files:
        print(f"No files matched: {directory / args.pattern}")
        return

    rows = [parse_log(p, target_sim_time=args.target) for p in files]

    fieldnames = [
        "file",
        "ok",
        "total_wall_s",
        "final_step",
        "final_sim_time",
        "final_dt",
        "avg_wall_per_step_s",
        "first_step_reaching_target",
        "estimated_wall_0_to_target_by_step_s",
        "estimated_wall_0_to_target_by_sim_time_s",
        "reason",
    ]

    out_csv = directory / args.csv
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    valid = [r for r in rows if r.get("ok") and r.get("total_wall_s") is not None]

    avg_step_values = [
        r["avg_wall_per_step_s"]
        for r in valid
        if r.get("avg_wall_per_step_s") is not None
    ]

    target_by_step_values = [
        r["estimated_wall_0_to_target_by_step_s"]
        for r in valid
        if r.get("estimated_wall_0_to_target_by_step_s") is not None
    ]

    target_by_sim_time_values = [
        r["estimated_wall_0_to_target_by_sim_time_s"]
        for r in valid
        if r.get("estimated_wall_0_to_target_by_sim_time_s") is not None
    ]

    print("\nPer-file summary")
    print("-" * 120)
    print(
        f"{'file':30s} {'wall_s':>10s} {'steps':>8s} {'final_t':>10s} "
        f"{'avg_s/step':>12s} {'step_to_target':>15s} {'est_0-target_s':>16s}"
    )
    print("-" * 120)

    for r in rows:
        if not r.get("ok"):
            print(f"{r['file']:30s} ERROR: {r.get('reason', '')}")
            continue

        print(
            f"{r['file']:30s} "
            f"{fmt(r.get('total_wall_s')):>10s} "
            f"{fmt(r.get('final_step')):>8s} "
            f"{fmt(r.get('final_sim_time')):>10s} "
            f"{fmt(r.get('avg_wall_per_step_s')):>12s} "
            f"{fmt(r.get('first_step_reaching_target')):>15s} "
            f"{fmt(r.get('estimated_wall_0_to_target_by_step_s')):>16s}"
        )

    print("-" * 120)
    print(f"CSV written to: {out_csv}")

    print("\nAggregate averages")
    print("-" * 120)

    print(f"Total files matched: {len(rows)}")
    print(f"Valid files with total wall time: {len(valid)}")

    if avg_step_values:
        print(
            f"Average wall time per step: {mean(avg_step_values):.6g} s/step "
            f"(n = {len(avg_step_values)})"
        )
    else:
        print("Average wall time per step: N/A (n = 0)")

    if target_by_step_values:
        print(
            f"Average estimated wall time from 0 to {args.target:g} simulation seconds "
            f"by step count: {mean(target_by_step_values):.6g} s "
            f"(n = {len(target_by_step_values)})"
        )
    else:
        print(
            f"Average estimated wall time from 0 to {args.target:g} simulation seconds "
            f"by step count: N/A (n = 0)"
        )

    if target_by_sim_time_values:
        print(
            f"Average estimated wall time from 0 to {args.target:g} simulation seconds "
            f"by simulation-time ratio: {mean(target_by_sim_time_values):.6g} s "
            f"(n = {len(target_by_sim_time_values)})"
        )
    else:
        print(
            f"Average estimated wall time from 0 to {args.target:g} simulation seconds "
            f"by simulation-time ratio: N/A (n = 0)"
        )

    skipped_target = [
        r["file"]
        for r in valid
        if r.get("first_step_reaching_target") is None
    ]
    if skipped_target:
        print(
            f"\nWarning: {len(skipped_target)} file(s) did not reach simulation time "
            f"{args.target:g}, so they were excluded from 0-target statistics:"
        )
        for name in skipped_target:
            print(f"  - {name}")


if __name__ == "__main__":
    main()