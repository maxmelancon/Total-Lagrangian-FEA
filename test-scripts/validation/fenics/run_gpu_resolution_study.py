#!/usr/bin/env python3
"""Run the CUDA FEAT10 explicit beam solver over all mesh resolutions.

The real-time factor uses the solver's summed CUDA-event Solve() time divided
by the requested physical simulation duration, matching the resolution-study
calculation in forward_beam_dynamic_v2.py.
"""

import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import subprocess


RESOLUTIONS = (0, 2, 4, 8, 16, 32)
DEFAULT_T_FINAL = 50e-3
SOLVE_TIME_PATTERN = re.compile(r"Total solve time:\s*([0-9.eE+-]+)\s*ms")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the C++ CUDA explicit beam solver at each mesh resolution."
    )
    parser.add_argument(
        "--resolution",
        choices=(*(str(r) for r in RESOLUTIONS), "all"),
        default="all",
        help="resolution to run, or 'all' (default: all)",
    )
    parser.add_argument(
        "--steps", type=int, default=5001,
        help="number of solver steps (default: 5001, matching the Python script)",
    )
    parser.add_argument(
        "--t-final", type=float, default=DEFAULT_T_FINAL,
        help="nominal simulation duration in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--dt", type=float,
        help="time step in seconds (default: t-final / (steps - 1), matching Python)",
    )
    parser.add_argument(
        "--executable",
        help="path to a built solver executable; default runs the Bazel target",
    )
    parser.add_argument(
        "--solver-args", default="",
        help="additional arguments passed to the solver, e.g. '--opt' or '--damp'",
    )
    parser.add_argument(
        "--csv",
        help="output CSV path (default: <project>/output/gpu_rtf_pairs_<steps>_steps.csv)",
    )
    return parser.parse_args()


def solver_command(args, project_root, resolution, dt):
    solver_args = [
        f"--res={resolution}",
        f"--steps={args.steps}",
        f"--dt={dt:.17g}",
        *shlex.split(args.solver_args),
    ]
    if args.executable:
        executable = Path(args.executable).expanduser()
        if not executable.is_absolute():
            executable = (project_root / executable).resolve()
        return [str(executable), *solver_args]
    return [
        "bazel", "run", "//lib_bin/beam_sag:test_feat10_explicit", "--",
        *solver_args,
    ]


def count_vector_dofs(project_root, resolution):
    node_file = (
        project_root
        / "data/meshes/T10/resolution"
        / f"beam_3x2x1_res{resolution}.1.node"
    )
    with node_file.open() as mesh_file:
        for line in mesh_file:
            line = line.partition("#")[0].strip()
            if line:
                # TetGen's first header field is the number of mesh nodes.
                return int(line.split()[0]) * 3
    raise ValueError(f"No node-count header found in mesh file: {node_file}")


def main():
    args = parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2 to match the Python timestep calculation")
    if args.t_final <= 0:
        raise SystemExit("--t-final must be positive")

    dt = args.dt if args.dt is not None else args.t_final / (args.steps - 1)
    if dt <= 0:
        raise SystemExit("--dt must be positive")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    resolutions = RESOLUTIONS if args.resolution == "all" else (int(args.resolution),)

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
    else:
        csv_path = project_root / "output" / f"gpu_rtf_pairs_{args.steps}_steps.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for resolution in resolutions:
        command = solver_command(args, project_root, resolution, dt)
        print(f"\nRunning GPU solver at resolution {resolution}:")
        print("  " + shlex.join(command))
        result = subprocess.run(
            command,
            cwd=project_root,
            env={
                **os.environ,
                "CC": "/usr/bin/gcc",
                "CXX": "/usr/bin/g++",
                "BUILD_WORKSPACE_DIRECTORY": str(project_root),
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.returncode != 0:
            raise SystemExit(
                f"Solver failed at resolution {resolution} (exit code {result.returncode}); "
                "no complete study CSV was written."
            )

        matches = SOLVE_TIME_PATTERN.findall(result.stdout)
        if not matches:
            raise SystemExit(
                f"Could not find 'Total solve time' in solver output at resolution {resolution}."
            )
        solve_time_seconds = float(matches[-1]) / 1000.0
        real_time_factor = solve_time_seconds / args.t_final
        dof_count = count_vector_dofs(project_root, resolution)
        rows.append({
            "resolution": resolution,
            "dof_count": dof_count,
            "real_time_factor": real_time_factor,
            "solve_time_seconds": solve_time_seconds,
            "simulation_time_seconds": args.t_final,
            "steps": args.steps,
            "dt_seconds": dt,
        })
        print(f"  GPU solve time: {solve_time_seconds:.6e} s")
        print(f"  Total vector DOFs: {dof_count}")
        print(f"  Real-time factor: {real_time_factor:.6e}")

    fieldnames = [
        "resolution",
        "dof_count",
        "real_time_factor",
        "solve_time_seconds",
        "simulation_time_seconds",
        "steps",
        "dt_seconds",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote resolution study CSV: {csv_path}")


if __name__ == "__main__":
    main()
