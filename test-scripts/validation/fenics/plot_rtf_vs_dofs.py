#!/usr/bin/env python3
"""Plot real-time factor against DOF count from existing solver-study CSVs."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt


GPU_PROJECT = Path(__file__).resolve().parents[3]
CPU_PROJECT = GPU_PROJECT.parents[1] / "Total-Lagrangian-FEA"
DEFAULT_CSVS = (
    CPU_PROJECT / "output/fenics_rtf_pairs_5001_steps.csv",
    GPU_PROJECT / "output/solver_comparison/beam_3x2x1/gpu_rtf_pairs_5001_steps.csv",
)
DEFAULT_OUTPUT = GPU_PROJECT / "output" / "beam_3x2x1_rtf_vs_dof.png"


def read_study_csv(path, default_solver=None, cpu_cores_override=None):
    """Read combined study rows or a per-solver study CSV."""
    points = []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header in {path}")
        for raw_row in reader:
            row = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in raw_row.items()
            }
            resolution = int(row["resolution"])
            dofs = int(float(row["dof_count"]))

            solve_seconds = row.get("solve_time_seconds", "")
            simulation_seconds = row.get("simulation_time_seconds", "")
            if solve_seconds and simulation_seconds:
                rtf = float(solve_seconds) / float(simulation_seconds)
            else:
                rtf_text = row.get("real_time_factor", row.get("rtf", ""))
                if not rtf_text:
                    raise ValueError(
                        f"{path} needs real_time_factor or solve_time_seconds plus "
                        "simulation_time_seconds"
                    )
                rtf = float(rtf_text)

            solver = row.get("solver") or default_solver or path.stem
            if solver.lower().startswith("fenics"):
                solver = "FEniCS CPU"
            elif solver.lower() in ("gpu", "cuda") or "cuda" in solver.lower():
                solver = "C++ CUDA"
            core_value = row.get("cpu_cores_used", "")
            if not core_value and cpu_cores_override is not None and solver == "FEniCS CPU":
                core_value = str(cpu_cores_override)
            points.append({
                "solver": solver,
                "resolution": resolution,
                "dof_count": dofs,
                "real_time_factor": rtf,
                "cpu_cores_used": int(core_value) if core_value else None,
            })

    if not points:
        raise ValueError(f"No study rows found in {path}")
    return points


def plot_rtf(points, output_path):
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    styles = {"FEniCS CPU": "tab:red", "C++ CUDA": "tab:blue"}
    groups = list(dict.fromkeys(
        (point["solver"], point["cpu_cores_used"] if point["solver"] == "FEniCS CPU" else None)
        for point in points
    ))
    cpu_core_counts = sorted({
        point["cpu_cores_used"] for point in points
        if point["solver"] == "FEniCS CPU" and point["cpu_cores_used"] is not None
    })
    for solver, cpu_cores in groups:
        label = solver
        if solver == "FEniCS CPU" and cpu_cores is not None:
            label = f"FEniCS CPU ({cpu_cores} cores)"
        series = sorted(
            (point["dof_count"], point["real_time_factor"], point["resolution"])
            for point in points
            if point["solver"] == solver
            and (point["cpu_cores_used"] if solver == "FEniCS CPU" else None) == cpu_cores
        )
        dofs, rtfs, _ = zip(*series)
        axis.plot(
            dofs, rtfs, marker="o", linewidth=1.8,
            color=styles.get(solver), label=label,
        )
        for dof, rtf, resolution in series:
            axis.annotate(
                str(resolution), (dof, rtf), xytext=(5, 5),
                textcoords="offset points", fontsize=8,
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Total vector degrees of freedom")
    axis.set_ylabel("Real-time factor (solve time / simulated time)")
    if len(cpu_core_counts) == 1:
        title = f"Real-Time Factor vs. DOF (CPU: {cpu_core_counts[0]} cores)"
    else:
        title = "Real-Time Factor vs. DOF"
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote log-log RTF vs DOF plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files", nargs="*", type=Path,
        help="combined CSV or per-solver study CSVs; defaults to the latest core-tagged workflow CSV when available",
    )
    parser.add_argument("--output", type=Path, help="plot path; default name includes CPU core count when available")
    parser.add_argument("--cpu-cores", type=int, help="core count to label when source CSVs lack it")
    args = parser.parse_args()

    ranked_csvs = sorted(
        (GPU_PROJECT / "output/solver_comparison/beam_3x2x1").glob(
            "beam_3x2x1_rtf_vs_dof_*_steps_cpu*_cores.csv"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    csv_files = args.csv_files or ([ranked_csvs[-1]] if ranked_csvs else list(DEFAULT_CSVS))
    inferred_cores = args.cpu_cores
    if inferred_cores is None:
        for path in csv_files:
            match = re.search(r"cpu(\d+)_cores", path.name)
            if match:
                inferred_cores = int(match.group(1))
                break
    points = []
    for path in csv_files:
        path = path.expanduser().resolve()
        if not path.is_file():
            parser.error(f"CSV not found: {path}")
        default_solver = None
        if "fenics" in path.name.lower():
            default_solver = "FEniCS CPU"
        elif "gpu" in path.name.lower() or "cuda" in path.name.lower():
            default_solver = "C++ CUDA"
        points.extend(read_study_csv(path, default_solver, inferred_cores))

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    elif inferred_cores is not None:
        output_path = (
            GPU_PROJECT / "output" / f"beam_3x2x1_rtf_vs_dof_cpu{inferred_cores}_cores.png"
        ).resolve()
    else:
        output_path = DEFAULT_OUTPUT.resolve()
    plot_rtf(points, output_path)


if __name__ == "__main__":
    main()
