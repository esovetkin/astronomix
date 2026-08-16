"""
Aggregate weak-scaling NPZ results into paper-style LaTeX tables.

Reads the ``weak_<setup>_pallas_<tag>_G*.npz`` files written by the weak
scaling drivers and prints, per setup, the table used in the methods paper
(GPUs, nodes, global grid, cells, runtime, efficiency, Mcell/s/GPU), both as
LaTeX rows and as a plain-text table.  Efficiency is T(1)/T(G) with the
1-GPU rung as baseline (ideal = 100%).

    python examples/scripts/scaling/make_weak_tables.py --tag gh200 \
        --gpus-per-node 4
"""

# general
import argparse
import glob
import os

# third-party
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "weak_scaling")

SETUPS = ["hydro", "mhd", "selfgrav"]


def _fmt_cells(n):
    exponent = int(np.floor(np.log10(n)))
    mantissa = n / 10 ** exponent
    return f"${mantissa:.1f}\\cdot10^{{{exponent}}}$"


def _fmt_grid(cells):
    nx, ny, nz = cells
    if nx == ny == nz:
        return f"${nx}^{{3}}$"
    if ny == nz:
        return f"${nx}\\times{ny}^{{2}}$"
    return f"${nx}\\times{ny}\\times{nz}$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="gh200")
    parser.add_argument("--gpus-per-node", type=int, default=4)
    args = parser.parse_args()

    for setup in SETUPS:
        pattern = os.path.join(
            DATA_DIR, f"weak_{setup}_pallas_{args.tag}_G*.npz"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[{setup}] no results matching {pattern}")
            continue

        rows = []
        for path in files:
            data = np.load(path, allow_pickle=True)
            G = int(data["G"])
            runtime = float(data["runtime"])
            cells = [int(c) for c in data["global_cells"]]
            total = int(data["total_cells"])
            # Recompute cell updates/s/GPU from the raw entries (early runs
            # recorded cells_per_s_per_gpu without the timestep factor).
            iters = int(data["iterations"])
            throughput = total * iters / runtime / G
            rows.append((G, cells, total, runtime, throughput))
        rows.sort(key=lambda r: r[0])

        baseline = next((r[3] for r in rows if r[0] == 1), None)

        print(f"\n===== {setup} (FD Pallas, {args.tag}) =====")
        print(f"{'GPUs':>5} {'Nodes':>5} {'Global grid':>18} {'Cells':>10} "
              f"{'Runtime[s]':>11} {'Eff.[%]':>8} {'Mcell/s/GPU':>12}")
        latex = []
        for G, cells, total, runtime, throughput in rows:
            nodes = max(1, G // args.gpus_per_node)
            eff = 100.0 * baseline / runtime if baseline else float("nan")
            print(f"{G:>5} {nodes:>5} "
                  f"{'x'.join(str(c) for c in cells):>18} {total:>10.2e} "
                  f"{runtime:>11.2f} {eff:>8.1f} {throughput / 1e6:>12.0f}")
            latex.append(
                f"        {G} & {nodes} & {_fmt_grid(cells)} & "
                f"{_fmt_cells(total)} & {runtime:.2f} & {eff:5.1f} & "
                f"{throughput / 1e6:.0f} \\\\"
            )
        print("\n% LaTeX rows:")
        print("\n".join(latex))


if __name__ == "__main__":
    main()
