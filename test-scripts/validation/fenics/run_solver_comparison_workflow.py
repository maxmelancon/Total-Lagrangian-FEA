#!/usr/bin/env python3
"""Run the fixed beam CPU/GPU comparisons and create both plots."""

import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt


WORKSPACE = Path(__file__).resolve().parents[5]
GPU_PROJECT = Path(__file__).resolve().parents[3]
CPU_PROJECT = WORKSPACE / "Total-Lagrangian-FEA"
CPU_SOLVER = CPU_PROJECT / "test-scripts/validation/fenics/forward_beam_dynamic_v2.py"
GPU_STUDY_SCRIPT = GPU_PROJECT / "test-scripts/validation/fenics/run_gpu_resolution_study.py"
GPU_TARGET = "//lib_bin/beam_sag:test_feat10_explicit"
FENICS_ENV = Path(os.environ.get(
    "FENICS_ENV_PREFIX", "/home/maxmelancon/miniconda3/envs/fenicsx"
))
FENICS_PYTHON = FENICS_ENV / "bin/python"
FENICS_MPIEXEC = FENICS_ENV / "bin/mpiexec"
MESH_NAME = "beam_3x2x1"
TRACKED_RAW_NODE_ID = 24
T_FINAL = 0.05
DEFAULT_STEPS = 5001
GPU_TIME_RE = re.compile(r"Total solve time:\s*([0-9.eE+-]+)\s*ms")


def available_physical_cores():
    """Count physical cores available to this process on Linux."""
    allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    cores = set()
    processor = physical_id = core_id = None
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in (*cpuinfo, ""):
                if not line.strip():
                    if processor is not None and (allowed is None or processor in allowed):
                        if physical_id is not None and core_id is not None:
                            cores.add((physical_id, core_id))
                    processor = physical_id = core_id = None
                    continue
                key, separator, value = line.partition(":")
                if not separator:
                    continue
                key, value = key.strip(), value.strip()
                if key == "processor":
                    processor = int(value)
                elif key == "physical id":
                    physical_id = int(value)
                elif key == "core id":
                    core_id = int(value)
    except OSError:
        pass
    return len(cores) or (os.cpu_count() or 1)


def fenics_mpi_command(args, *solver_args):
    return [
        FENICS_MPIEXEC, "-n", str(args.cpu_ranks), "-bind-to", "core",
        FENICS_PYTHON, CPU_SOLVER, *solver_args,
    ]


def fenics_environment():
    return {
        **os.environ,
        "PATH": f"{FENICS_ENV / 'bin'}:{os.environ.get('PATH', '')}",
        # Use MPI ranks across physical cores without nested BLAS/OpenMP threads.
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def run(command, cwd, env=None):
    command = [str(part) for part in command]
    print("\n$ " + shlex.join(command), flush=True)
    child_env = {**(os.environ if env is None else env), "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command, cwd=cwd, env=child_env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Command failed with exit code {returncode}")
    return "".join(output)


def output_dir_for(args):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def node_data():
    node_file = GPU_PROJECT / "data/meshes/T10/resolution/beam_3x2x1_res0.1.node"
    nodes = {}
    with node_file.open() as mesh_file:
        next(mesh_file)
        for line in mesh_file:
            line = line.partition("#")[0].strip()
            if line:
                fields = line.split()
                nodes[int(fields[0])] = tuple(float(v) for v in fields[1:4])
    if TRACKED_RAW_NODE_ID not in nodes:
        raise ValueError(f"Node {TRACKED_RAW_NODE_ID} missing from {node_file}")
    return nodes[TRACKED_RAW_NODE_ID]


def read_history(path):
    with path.open(newline="") as csv_file:
        records = csv.DictReader(
            line for line in csv_file if not line.lstrip().startswith("#")
        )
        rows = [
            (int(row["step"]), float(row["x_position"]),
             float(row["y_position"]), float(row["z_position"]))
            for row in records
        ]
    if not rows:
        raise ValueError(f"No response data in {path}")
    return rows


def plot_response(cpu_csv, gpu_csv, initial_position, dt, cpu_cores, output_path):
    histories = ((read_history(cpu_csv), f"FEniCS CPU ({cpu_cores} cores)", "tab:red"),
                 (read_history(gpu_csv), "C++ CUDA", "tab:blue"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for component, axis in enumerate(axes):
        for history, label, color in histories:
            times = [(row[0] + 1) * dt for row in history]
            displacement = [row[component + 1] - initial_position[component] for row in history]
            axis.plot(times, displacement, color=color, label=label)
        axis.set_title(("x", "y", "z")[component])
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Displacement (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        f"Tracked node {TRACKED_RAW_NODE_ID} displacement — {MESH_NAME} "
        f"(CPU: {cpu_cores} cores)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote response plot: {output_path}")


def read_rtf_csv(path, label, cpu_cores_used=None):
    rows = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append({
                "solver": label,
                "resolution": int(row["resolution"]),
                "dof_count": int(row["dof_count"]),
                "real_time_factor": float(row["real_time_factor"]),
                "cpu_cores_used": cpu_cores_used if label == "FEniCS CPU" else "",
            })
    return rows


def plot_rtf(rows, cpu_cores, output_path):
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    display_labels = {
        "FEniCS CPU": f"FEniCS CPU ({cpu_cores} cores)",
        "C++ CUDA": "C++ CUDA",
    }
    for label, color in (("FEniCS CPU", "tab:red"), ("C++ CUDA", "tab:blue")):
        points = sorted(
            (row["dof_count"], row["real_time_factor"], row["resolution"])
            for row in rows if row["solver"] == label
        )
        dofs, rtfs, _ = zip(*points)
        axis.plot(dofs, rtfs, marker="o", color=color, label=display_labels[label])
        for dof, rtf, resolution in points:
            axis.annotate(str(resolution), (dof, rtf), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Total vector degrees of freedom")
    axis.set_ylabel("Real-time factor (solve time / simulated time)")
    axis.set_title(f"RTF vs DOF — {MESH_NAME} (CPU: {cpu_cores} cores)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote RTF vs DOF plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    max_cpu_cores = available_physical_cores()
    parser.add_argument(
        "--cpu-ranks", type=int, default=max_cpu_cores,
        help=f"FEniCS MPI processes / bound CPU cores (default: {max_cpu_cores})",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=GPU_PROJECT / "output" / "solver_comparison" / MESH_NAME,
    )
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.cpu_ranks < 1:
        parser.error("--cpu-ranks must be at least 1")
    if args.cpu_ranks > max_cpu_cores:
        parser.error(f"--cpu-ranks cannot exceed {max_cpu_cores} available physical cores")
    if not FENICS_MPIEXEC.is_file() or not FENICS_PYTHON.is_file():
        parser.error(
            f"FEniCS MPI/Python executables not found under {FENICS_ENV}; "
            "set FENICS_ENV_PREFIX to the fenicsx environment path"
        )

    output_dir = output_dir_for(args)
    dt = T_FINAL / (args.steps - 1)
    run_tag = f"cpu{args.cpu_ranks}_cores"
    cpu_response_source = CPU_PROJECT / "output" / (
        f"forward_beam_res0_steps{args.steps}_t50ms_load10000N.csv"
    )
    cpu_response_csv = output_dir / f"{MESH_NAME}_fenics_res0_response_{run_tag}.csv"
    gpu_response_csv = output_dir / f"{MESH_NAME}_gpu_res0_response_{run_tag}.csv"

    # Response CSVs first: FEniCS writes its standard output file; Bazel writes GPU CSV here.
    print(f"FEniCS CPU MPI ranks: {args.cpu_ranks}")
    cpu_env = fenics_environment()
    run(
        fenics_mpi_command(args, "0", "true", args.steps),
        CPU_PROJECT,
        cpu_env,
    )
    shutil.copy2(cpu_response_source, cpu_response_csv)
    gpu_env = {
        **os.environ,
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "BUILD_WORKSPACE_DIRECTORY": str(GPU_PROJECT),
    }
    run([
        "bazel", "run", GPU_TARGET, "--",
        "--res=0", f"--steps={args.steps}", f"--dt={dt:.17g}",
        f"--csv={gpu_response_csv}",
    ], GPU_PROJECT, gpu_env)
    plot_response(
        cpu_response_csv, gpu_response_csv, node_data(), dt, args.cpu_ranks,
        output_dir / f"{MESH_NAME}_res0_xyz_response_{run_tag}.png",
    )

    # Run the existing resolution studies; they write one CSV per solver.
    cpu_rtf_source = CPU_PROJECT / "output" / f"fenics_rtf_pairs_{args.steps}_steps.csv"
    cpu_rtf_csv = output_dir / f"fenics_rtf_pairs_{args.steps}_steps_{run_tag}.csv"
    gpu_rtf_csv = output_dir / f"gpu_rtf_pairs_{args.steps}_steps_{run_tag}.csv"
    run(
        fenics_mpi_command(args, "all", "false", args.steps),
        CPU_PROJECT,
        cpu_env,
    )
    shutil.copy2(cpu_rtf_source, cpu_rtf_csv)
    run([
        sys.executable, GPU_STUDY_SCRIPT,
        "--steps", args.steps,
        "--t-final", T_FINAL,
        "--csv", gpu_rtf_csv,
    ], GPU_PROJECT)

    rows = read_rtf_csv(cpu_rtf_csv, "FEniCS CPU", cpu_cores_used=args.cpu_ranks)
    rows.extend(read_rtf_csv(gpu_rtf_csv, "C++ CUDA"))
    combined_csv = output_dir / f"{MESH_NAME}_rtf_vs_dof_{args.steps}_steps_{run_tag}.csv"
    with combined_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "solver", "resolution", "dof_count", "real_time_factor",
                "cpu_cores_used",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote combined RTF/DOF CSV: {combined_csv}")
    plot_rtf(
        rows, args.cpu_ranks,
        output_dir / f"{MESH_NAME}_rtf_vs_dof_{run_tag}.png",
    )
    print(f"\nWorkflow complete. Outputs are in {output_dir}")


if __name__ == "__main__":
    main()
