"""FD-Pallas single- vs double-precision per-iteration runtime (A100 vs H100).

Single-panel companion to ``alfven_convergence.py``: how much double precision
actually costs the FD (Pallas) MHD solver per timestep, and how that cost differs
between GPU generations.  The dp/sp gap is NOT a pure hardware-ratio effect --
the f32 kernel neither spills nor uses the same WENO-weight algebra as the f64
one -- so it is measured rather than derived.

The sweep runs one precision per process (``jax_enable_x64`` is global and must
be set before the first array is built), so a full node needs two invocations:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_spdp_runtime.py --dp
    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_spdp_runtime.py --sp

each writing ``pytests/mhd/data/astronomix/<node>/alfven_wave3D_spdp_runtime_{dp,sp}.npz``.
The node tag is derived from the live device (a100/h100/h200/...) so a run
cannot be silently mislabelled; override with ``--tag <name>`` only if you must.
Both are driven for you by ``generate_alfven_data.py``.

Once both precisions exist on a node, render the figures on the CPU (all nodes
found on disk are drawn together -- comparing them is the point):

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_spdp_runtime.py --plot

    pytests/mhd/figures/alfven_wave3D_spdp_runtime.svg   (per-iteration runtime vs N)
    pytests/mhd/figures/alfven_wave3D_spdp_speedup.svg   (dp/sp ratio vs N, per node)

The speedup figure is the one that compares nodes honestly: absolute runtimes
are hardware-bound and belong on separate axes, but the dp/sp ratio divides that
out, so A100 and H100 curves can share a panel.
"""

# general
# NOTE: os/sys must come first — --plot decides whether we grab a GPU via
# autocvd or stay on the CPU, and both must happen before jax is imported.
import os
import sys
from pathlib import Path

PLOT_ONLY = "--plot" in sys.argv

# ==== GPU selection ====
if PLOT_ONLY:
    # Plotting touches no jax ops; keep jax off the GPUs entirely.
    os.environ["JAX_PLATFORMS"] = "cpu"
elif os.environ.get("CUDA_VISIBLE_DEVICES"):
    # A parent (generate_alfven_data.py) already picked the GPU via autocvd and
    # exported it; inherit it so both precisions are timed on the same device.
    pass
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import jax

if "--sp" in sys.argv and "--dp" in sys.argv:
    raise SystemExit("pick exactly one of --sp / --dp")
PRECISION = "sp" if "--sp" in sys.argv else "dp"
jax.config.update("jax_enable_x64", PRECISION == "dp")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "pytests"))

from astronomix import SimulationConfig, SnapshotSettings
from astronomix.option_classes.simulation_config import StaticFloatVector
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave

from _benchmark_utils import BenchmarkSpec, _run_simulation, node_tag

# Per-node storage, matching alfven_convergence.py: runtimes are only
# comparable within one GPU generation.
DATA_ROOT = _REPO / "pytests" / "mhd" / "data" / "astronomix"
FIG_DIR = _REPO / "pytests" / "mhd" / "figures"

N_VALUES = [8, 16, 32, 64, 128]

# Fixed step count: this figure is about cost per iteration, so the adaptive
# timestep (which differs between precisions) must not enter the comparison.
STEPS = 20


def _node_tag() -> str:
    """Node tag from the live device, or an explicit ``--tag`` override."""
    for i, flag in enumerate(sys.argv):
        if flag == "--tag" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].lower()
    return node_tag()


def _npz_path(precision: str, tag: str) -> Path:
    return DATA_ROOT / tag / f"alfven_wave3D_spdp_runtime_{precision}.npz"


def _nodes_on_disk() -> list:
    """Every node tag that has at least one sp/dp NPZ."""
    if not DATA_ROOT.exists():
        return []
    return sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and any(d.glob("alfven_wave3D_spdp_runtime_*.npz"))
    )


def run() -> None:
    tag = _node_tag()
    device = jax.devices()[0].device_kind
    print(f"[spdp] {PRECISION} on {device} (tag={tag}), {STEPS} fixed steps", flush=True)

    spec = BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            box_size=StaticFloatVector(3.0, 1.5, 1.5),
            mhd=True,
            dimensionality=3,
            progress_bar=False,
            print_elapsed_time=True,
            return_snapshots=True,
            snapshot_settings=SnapshotSettings(return_final_state=True),
        ),
        cfl=1.5,
    )

    per_iter_ms, runtimes, iterations = [], [], []
    for N in N_VALUES:
        # robust_timing: a single 20-step run at small N lasts only ~0.04 s, and
        # a sporadic ~0.1 s one-off cost inside the timed region would show up as
        # a 2-4x error in ms/iter (see _run_simulation).  Repeat-and-take-min.
        result, *_ = _run_simulation(
            spec, N, setup_cp_alfven_wave, num_gpus=1, num_timesteps=STEPS,
            robust_timing=True,
        )
        rt = float(result.runtime)
        iters = int(result.num_iterations)
        ms = 1000.0 * rt / iters
        per_iter_ms.append(ms)
        runtimes.append(rt)
        iterations.append(iters)
        print(f"[spdp] {PRECISION} N={N:4d}: {ms:8.3f} ms/iter "
              f"(runtime={rt:.3f}s, iters={iters})", flush=True)

    out = _npz_path(PRECISION, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        N_values=np.asarray(N_VALUES),
        per_iter_ms=np.asarray(per_iter_ms),
        runtimes=np.asarray(runtimes),
        iterations=np.asarray(iterations),
        precision=PRECISION,
        node=tag,
        device_kind=device,
    )
    print(f"[spdp] wrote {out}", flush=True)


def plot() -> None:
    """Single panel: per-iteration runtime vs N, both precisions, every node.

    Nodes are discovered from disk rather than hard-coded, so a sweep on new
    hardware (h200, gh200, ...) joins the figure with no code change.
    """
    # Node identity -> colour; precision -> linestyle/marker.  Keeps the curves
    # readable as per-node pairs rather than N unrelated lines.
    palette = ("tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple")
    nodes = _nodes_on_disk()
    node_styles = {n: (palette[i % len(palette)], n.upper()) for i, n in enumerate(nodes)}
    prec_styles = {"dp": ("-", "o", "double"), "sp": ("--", "s", "single")}

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    found = []
    for node, (colour, node_label) in node_styles.items():
        for precision, (ls, marker, prec_label) in prec_styles.items():
            path = _npz_path(precision, node)
            if not path.exists():
                print(f"[spdp] missing, skipping curve: {node}/{path.name}")
                continue
            d = np.load(path)
            ax.loglog(
                d["N_values"], d["per_iter_ms"],
                color=colour, linestyle=ls, marker=marker, linewidth=2,
                label=f"{node_label} {prec_label}",
            )
            found.append((node, precision, d))

    # Annotate the measured dp/sp ratio at the largest common N per node —
    # the number this figure exists to communicate.
    for i, (node, (colour, node_label)) in enumerate(node_styles.items()):
        have = {p: d for n, p, d in found if n == node}
        if {"dp", "sp"} <= set(have):
            N_dp, N_sp = have["dp"]["N_values"], have["sp"]["N_values"]
            common = sorted(set(N_dp.tolist()) & set(N_sp.tolist()))
            if not common:
                continue
            N = common[-1]
            dp_ms = float(have["dp"]["per_iter_ms"][list(N_dp).index(N)])
            sp_ms = float(have["sp"]["per_iter_ms"][list(N_sp).index(N)])
            ax.annotate(
                f"{node_label}: dp/sp = {dp_ms / sp_ms:.1f}×",
                xy=(N, dp_ms), xytext=(0.98, 0.30 - 0.07 * i),
                textcoords="axes fraction", ha="right", color=colour, fontsize=10,
            )

    ax.set_xlabel("N (cells per side; grid 2N × N × N)")
    ax.set_ylabel("runtime per iteration [ms]")
    ax.set_title("FD (Pallas) MHD: precision cost per iteration")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "alfven_wave3D_spdp_runtime.svg"
    fig.savefig(out)
    print(f"[spdp] wrote {out}")
    if not found:
        print("[spdp] WARNING: no data found — run --dp/--sp on each node first")


def plot_speedup() -> None:
    """dp/sp speedup vs N, one curve per node.

    How much faster single precision is than double for the same FD (Pallas)
    step. Being a ratio of two runs of the same solver on the same device, it
    cancels anything that hits both precisions equally (clocks, occupancy,
    machine), which is what makes A100 and H100 comparable on one axis even
    though their absolute runtimes are not.

    The ratio is measured, not derived: it is NOT the fp64:fp32 hardware ratio,
    because the two kernels are not the same computation (the f32 path keeps the
    classic WENO-weight algebra and does not spill, the f64 path does neither).
    """
    palette = ("tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple")
    nodes = _nodes_on_disk()

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    drawn = 0
    all_N: set = set()
    all_ratios: list = []
    for i, node in enumerate(nodes):
        dp_path, sp_path = _npz_path("dp", node), _npz_path("sp", node)
        if not (dp_path.exists() and sp_path.exists()):
            print(f"[spdp] {node}: need both precisions for a speedup curve, skipping")
            continue
        dp, sp = np.load(dp_path), np.load(sp_path)
        dp_ms = {int(n): float(v) for n, v in zip(dp["N_values"], dp["per_iter_ms"])}
        sp_ms = {int(n): float(v) for n, v in zip(sp["N_values"], sp["per_iter_ms"])}
        common = sorted(set(dp_ms) & set(sp_ms))
        if not common:
            continue
        ratio = [dp_ms[n] / sp_ms[n] for n in common]
        colour = palette[i % len(palette)]
        ax.plot(common, ratio, color=colour, marker="o", linewidth=2, label=node.upper())
        for n, r in zip(common, ratio):
            ax.annotate(f"{r:.1f}×", xy=(n, r), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color=colour)
        all_N.update(common)
        all_ratios.extend(ratio)
        drawn += 1

    # 1x = no benefit from single precision; the curves are only meaningful above it.
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)

    # Label the axis with the actual cell counts (8, 16, ...) rather than the
    # 2^3 exponents a log2 axis defaults to -- N is a grid size to look up, not
    # a power to read off.
    if all_N:
        ticks = sorted(all_N)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticklabels([str(n) for n in ticks])

    # Headroom: the per-point "5.8×" labels sit above their markers and would
    # otherwise collide with (or spill through) the top frame.
    if all_ratios:
        lo, hi = min(all_ratios + [1.0]), max(all_ratios)
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.10 * span, hi + 0.22 * span)

    ax.set_xlabel("N (cells per side; grid 2N × N × N)")
    ax.set_ylabel("dp / sp runtime per iteration")
    ax.set_title("FD (Pallas) MHD: single-precision speedup")
    ax.grid(True, which="both", alpha=0.3)
    if drawn:
        ax.legend(frameon=False, title="GPU")
    fig.tight_layout()

    out = FIG_DIR / "alfven_wave3D_spdp_speedup.svg"
    fig.savefig(out)
    print(f"[spdp] wrote {out}")
    if not drawn:
        print("[spdp] WARNING: no node has both precisions — no speedup curve drawn")


if __name__ == "__main__":
    if PLOT_ONLY:
        plot()
        plot_speedup()
    else:
        run()
