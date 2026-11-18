#!/usr/bin/env python3
"""Compare CPU/GPU RTF and resolution-0 responses."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
OUTPUT_DIR = PROJECT_DIR / "output/solver_comparison/beam_3x2x1"
RTF_FILES = {
    8: OUTPUT_DIR / "fenics_rtf_pairs_5001_steps_cpu8_cores.csv",
    16: OUTPUT_DIR / "fenics_rtf_pairs_5001_steps_cpu16_cores.csv",
}
GPU_RTF_FILE = OUTPUT_DIR / "gpu_rtf_pairs_5001_steps_cpu16_cores.csv"
LEGACY_RTF_FILE = OUTPUT_DIR / "beam_3x2x1_rtf_vs_dof_5001_steps.csv"
RESPONSE_FILES = {
    1: OUTPUT_DIR / "beam_3x2x1_fenics_res0_response_cpu1_cores.csv",
    8: OUTPUT_DIR / "beam_3x2x1_fenics_res0_response_cpu8_cores.csv",
    16: OUTPUT_DIR / "beam_3x2x1_fenics_res0_response_cpu16_cores.csv",
}
GPU_RESPONSE_FILE = OUTPUT_DIR / "beam_3x2x1_gpu_res0_response_cpu16_cores.csv"
SIMULATION_TIME = 0.05


def load_rtf_rows():
    rows = []
    with LEGACY_RTF_FILE.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["solver"].strip().lower().startswith("fenics"):
                rows.append({
                    "cpu_cores_used": 1,
                    "resolution": int(row["resolution"]),
                    "dof_count": int(row["dof_count"]),
                    "real_time_factor": float(row["real_time_factor"]),
                    "run_settings": "earlier_serial_run",
                })
    for cores, path in RTF_FILES.items():
        with path.open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                rows.append({
                    "cpu_cores_used": cores,
                    "resolution": int(row["resolution"]),
                    "dof_count": int(row["dof_count"]),
                    "real_time_factor": float(row["real_time_factor"]),
                    "run_settings": "standardized_ksp",
                })
    with GPU_RTF_FILE.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append({
                "cpu_cores_used": "GPU",
                "resolution": int(row["resolution"]),
                "dof_count": int(row["dof_count"]),
                "real_time_factor": float(row["real_time_factor"]),
                "run_settings": "CUDA, from 16-core workflow",
            })
    core_order = {1: 1, 8: 8, 16: 16, "GPU": 32}
    return sorted(rows, key=lambda row: (core_order[row["cpu_cores_used"]], row["resolution"]))


def load_response(path):
    with path.open() as csv_file:
        rows = list(csv.DictReader(line for line in csv_file if not line.startswith("#")))
    step = np.asarray([int(row["step"]) for row in rows])
    positions = np.asarray([
        [float(row["x_position"]), float(row["y_position"]), float(row["z_position"])]
        for row in rows
    ])
    time = step * (SIMULATION_TIME / (len(step) - 1))
    # Remove each run's own starting position so the comparison measures motion.
    displacement_mm = (positions - positions[0]) * 1000.0
    return time, displacement_mm


def write_rtf_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["cpu_cores_used", "resolution", "dof_count", "real_time_factor", "run_settings"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_rtf(rows, output_path):
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    colors = {1: "tab:blue", 8: "tab:orange", 16: "tab:green", "GPU": "tab:red"}
    for cores in (1, 8, 16, "GPU"):
        series = sorted((row for row in rows if row["cpu_cores_used"] == cores), key=lambda row: row["dof_count"])
        axis.plot(
            [row["dof_count"] for row in series],
            [row["real_time_factor"] for row in series],
            marker="o", linewidth=2, color=colors[cores],
            label=("1 core (earlier run)" if cores == 1 else
                   "GPU (CUDA)" if cores == "GPU" else f"{cores} cores"),
        )
        for row in series:
            axis.annotate(str(row["resolution"]), (row["dof_count"], row["real_time_factor"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Total vector DOFs")
    axis.set_ylabel("Real time factor (wall time / simulated time)")
    axis.set_title("CPU and GPU RTF by DOF count\n1-core CPU data is from the earlier KSP configuration")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accuracy(output_path):
    responses = {cores: load_response(path) for cores, path in RESPONSE_FILES.items()}
    responses["GPU"] = load_response(GPU_RESPONSE_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    components = ("x", "y", "z")
    colors = {1: "tab:blue", 8: "tab:orange", 16: "tab:green", "GPU": "tab:red"}
    reference = responses[1][1]
    max_error_um = 0.0

    for component, name in enumerate(components):
        for cores in (1, 8, 16, "GPU"):
            time, displacement = responses[cores]
            axes[component].plot(time, displacement[:, component], color=colors[cores],
                                 linewidth=1.6,
                                 label=("GPU (CUDA)" if cores == "GPU" else
                                        f"{cores} core{'s' if cores != 1 else ''}"))
            max_error_um = max(max_error_um,
                               float(np.max(np.abs(displacement[:, component] - reference[:, component])) * 1000.0))
        axes[component].set_title(f"{name.upper()} displacement")
        axes[component].set_xlabel("Time (s)")
        axes[component].set_ylabel("Displacement (mm)")
        axes[component].grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle(
        "Resolution 0 CPU/GPU response comparison\n"
        f"Maximum deviation from 1-core CPU: {max_error_um:.3g} µm", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rtf_rows()
    rtf_csv = args.output_dir / "beam_3x2x1_cpu_gpu_rtf_5001_steps.csv"
    rtf_png = args.output_dir / "beam_3x2x1_cpu_gpu_rtf_vs_dof.png"
    accuracy_png = args.output_dir / "beam_3x2x1_cpu_gpu_accuracy_res0.png"
    write_rtf_csv(rows, rtf_csv)
    plot_rtf(rows, rtf_png)
    plot_accuracy(accuracy_png)
    print(f"Wrote {rtf_csv}")
    print(f"Wrote {rtf_png}")
    print(f"Wrote {accuracy_png}")


if __name__ == "__main__":
    main()
