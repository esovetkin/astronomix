#!/usr/bin/env python3
"""
AthenaPK 3D CP Alfvén wave convergence test.

Runs the circularly polarized Alfvén wave test (cpaw problem generator)
at multiple resolutions N, where the grid is 2N × N × N.  Results are
saved as numpy files so that AthenaPK can be added to the astronomix
convergence plot.

Wave setup matches the astronomix test:
  - Box: Lx=3, Ly=Lz=1.5, periodic boundaries
  - Propagation direction: (1,2,2)/3 (domain diagonal)
  - rho=1, p=0.1, B_par=1, B_perp=0.1, gamma=5/3
  - t_end=5 (five wave periods; analytic state returns to initial condition)

L1 error definition (matches astronomix "average over 8 primitive variables"):
  L1 = (err_rho + err_M1 + err_M2 + err_M3 + err_E + err_B1 + err_B2 + err_B3) / 8
  Since rho=1 everywhere in this test: err_M_i ≈ err_v_i (momentum ≈ velocity).
  err_rho and err_p (via err_E) are negligible for the CP Alfven wave.

Usage:
  python run_alfven_convergence.py /path/to/build/bin/athenaPK
  python run_alfven_convergence.py /path/to/athenaPK --N-values 8 16 32 64
  python run_alfven_convergence.py /path/to/athenaPK --output-dir results/

Output files (in --output-dir):
  athenapk_alfven_errors.npy       -- L1 errors, shape (len(N_values),)
  athenapk_alfven_runtimes.npy     -- wall-clock times in seconds
  athenapk_alfven_iterations.npy   -- cycle counts (Ncycle from cpaw-errors.dat)
  athenapk_alfven_N_values.npy     -- N values used
  athenapk_alfven_convergence.npz  -- all arrays in one archive
"""

import argparse
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default test parameters
# ---------------------------------------------------------------------------
DEFAULT_N_VALUES = [8, 16, 32, 64, 128]
T_END = 5.0
CFL = 0.3


def meshblock_for(N: int) -> tuple[int, int, int]:
    """
    Meshblock dimensions for grid (2N, N, N).
    Use the full grid for small N; cap at 32 per axis otherwise.
    """
    mb1 = 2 * N
    mb2 = N
    mb3 = N
    return mb1, mb2, mb3


def parse_cpaw_errors(filepath: Path) -> tuple[dict, int]:
    """
    Parse cpaw-errors.dat and return (per_var_errors, ncycle).

    File format (written by cpaw.cpp):
        # Nx1  Nx2  Nx3  Ncycle  RMS-Error  d  M1  M2  M3  E  B1c  B2c  B3c
    """
    with open(filepath) as fh:
        data_lines = [ln for ln in fh if not ln.startswith("#")]

    if not data_lines:
        raise RuntimeError(f"No data in {filepath}")

    cols = data_lines[-1].split()
    # columns: Nx1 Nx2 Nx3 Ncycle RMS d M1 M2 M3 E B1c B2c B3c
    ncycle = int(cols[3])
    errors = {
        "rms": float(cols[4]),
        "d":   float(cols[5]),
        "M1":  float(cols[6]),
        "M2":  float(cols[7]),
        "M3":  float(cols[8]),
        "E":   float(cols[9]),
        "B1c": float(cols[10]),
        "B2c": float(cols[11]),
        "B3c": float(cols[12]),
    }
    return errors, ncycle


def average_l1_error(errors: dict) -> float:
    """
    Average L1 error over 8 conserved-variable proxies.

    Matches the astronomix definition:
        (err_rho + err_vx + err_vy + err_vz + err_p + err_Bx + err_By + err_Bz) / 8

    Because rho=1 in this test, err_M_i = err_v_i exactly, and err_rho ≈ 0.
    The energy error err_E serves as a proxy for the pressure error.
    """
    return (
        errors["d"]
        + errors["M1"]
        + errors["M2"]
        + errors["M3"]
        + errors["E"]
        + errors["B1c"]
        + errors["B2c"]
        + errors["B3c"]
    ) / 8.0


def parse_throughput(stdout: str) -> float | None:
    """
    Extract Mzone-cycles/wallsecond from Parthenon stdout.
    Returns None if not found.
    """
    match = re.search(r"zone-cycles/wallsecond\s*=\s*([\d.eE+\-]+)", stdout)
    if match:
        return float(match.group(1))
    return None


def run_one(binary: str, input_file: Path, N: int, run_dir: str) -> tuple[dict, int, float]:
    """
    Run athenaPK for grid 2N×N×N and return (errors, ncycle, sim_walltime_s).

    sim_walltime_s is derived from Parthenon's zone-cycle throughput metric
    when available; otherwise falls back to Python-measured wall time (which
    includes subprocess startup overhead).
    """
    nx1, nx2, nx3 = 2 * N, N, N
    mb1, mb2, mb3 = meshblock_for(N)

    cmd = [
        binary, "-i", str(input_file),
        f"parthenon/mesh/nx1={nx1}",
        f"parthenon/mesh/nx2={nx2}",
        f"parthenon/mesh/nx3={nx3}",
        f"parthenon/meshblock/nx1={mb1}",
        f"parthenon/meshblock/nx2={mb2}",
        f"parthenon/meshblock/nx3={mb3}",
        f"parthenon/time/tlim={T_END}",
        f"parthenon/time/cfl={CFL}",
    ]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
    t_elapsed = time.perf_counter() - t_start

    if result.returncode != 0:
        print("--- stdout ---")
        print(result.stdout[-2000:])
        print("--- stderr ---")
        print(result.stderr[-2000:])
        raise RuntimeError(f"athenaPK failed for N={N} (exit {result.returncode})")

    errors, ncycle = parse_cpaw_errors(Path(run_dir) / "cpaw-errors.dat")

    # Prefer throughput-derived time to exclude Python/startup overhead
    throughput = parse_throughput(result.stdout)
    if throughput and throughput > 0:
        total_zone_cycles = nx1 * nx2 * nx3 * ncycle
        sim_time = total_zone_cycles / throughput
    else:
        sim_time = t_elapsed

    return errors, ncycle, sim_time


def find_input_file(script_dir: Path, user_input: str | None) -> Path:
    if user_input:
        p = Path(user_input).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = [
        script_dir / "cpaw_convergence.in",
        script_dir.parent / "inputs" / "cpaw_convergence.in",
        script_dir.parent / "inputs" / "cpaw.in",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "Could not find cpaw_convergence.in or cpaw.in. "
        "Pass --input /path/to/input.in explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AthenaPK CP Alfvén wave convergence test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("binary", help="Path to athenaPK executable")
    parser.add_argument("--input", default=None, help="Base input file (.in)")
    parser.add_argument(
        "--N-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        metavar="N",
        help="N values to test (grid is 2N×N×N for each N)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory in which to save numpy output files",
    )
    args = parser.parse_args()

    binary = str(Path(args.binary).resolve())
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"Binary not found: {binary}")

    script_dir = Path(__file__).parent
    input_file = find_input_file(script_dir, args.input)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = args.N_values
    l1_errors  = []
    runtimes   = []
    ncycles    = []

    print(f"AthenaPK CP Alfvén wave convergence")
    print(f"  binary    : {binary}")
    print(f"  input file: {input_file}")
    print(f"  N values  : {n_values}  (grids: {['%d×%d×%d' % (2*n,n,n) for n in n_values]})")
    print(f"  t_end={T_END}, CFL={CFL}, VL2+PLM+HLLD, GLM-MHD")
    print()

    for N in n_values:
        print(f"  N={N:4d}  grid={2*N}×{N}×{N}  meshblock={meshblock_for(N)}", end="", flush=True)

        with tempfile.TemporaryDirectory(prefix=f"athenapk_cpaw_N{N}_") as run_dir:
            errors, ncycle, sim_time = run_one(binary, input_file, N, run_dir)

        l1 = average_l1_error(errors)
        l1_errors.append(l1)
        runtimes.append(sim_time)
        ncycles.append(ncycle)

        print(f"  →  L1={l1:.3e}  t={sim_time:.2f}s  Ncycle={ncycle}")

    # -----------------------------------------------------------------------
    # Save numpy files
    # -----------------------------------------------------------------------
    arr_n      = np.array(n_values,    dtype=np.int64)
    arr_err    = np.array(l1_errors,   dtype=np.float64)
    arr_time   = np.array(runtimes,    dtype=np.float64)
    arr_cycles = np.array(ncycles,     dtype=np.int64)

    np.save(output_dir / "athenapk_alfven_N_values.npy",   arr_n)
    np.save(output_dir / "athenapk_alfven_errors.npy",     arr_err)
    np.save(output_dir / "athenapk_alfven_runtimes.npy",   arr_time)
    np.save(output_dir / "athenapk_alfven_iterations.npy", arr_cycles)
    np.savez(
        output_dir / "athenapk_alfven_convergence.npz",
        N_values=arr_n,
        l1_errors=arr_err,
        runtimes=arr_time,
        iterations=arr_cycles,
    )

    print(f"\nSaved to {output_dir}/")
    for fname in [
        "athenapk_alfven_N_values.npy",
        "athenapk_alfven_errors.npy",
        "athenapk_alfven_runtimes.npy",
        "athenapk_alfven_iterations.npy",
        "athenapk_alfven_convergence.npz",
    ]:
        print(f"  {fname}")

    # -----------------------------------------------------------------------
    # Print convergence rates
    # -----------------------------------------------------------------------
    if len(n_values) >= 2:
        print("\nConvergence rates (log-log slope between consecutive resolutions):")
        for i in range(1, len(n_values)):
            slope = np.log(arr_err[i] / arr_err[i - 1]) / np.log(n_values[i] / n_values[i - 1])
            print(f"  N={n_values[i-1]:3d} → {n_values[i]:3d}: {slope:+.2f}  (expect ≈ -2 for PLM/VL2)")


if __name__ == "__main__":
    main()
