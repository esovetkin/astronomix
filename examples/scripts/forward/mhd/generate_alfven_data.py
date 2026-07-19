"""Generate ALL astronomix Alfvén benchmark data for this GPU, in one command.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/generate_alfven_data.py

Runs, in order:

1. ``alfven_convergence.py``       — the dp convergence + runtime sweep for all
                                     three backends (FV JAX, FD JAX, FD Pallas)
2. ``alfven_spdp_runtime.py --dp`` — per-iteration cost, double precision
3. ``alfven_spdp_runtime.py --sp`` — per-iteration cost, single precision
4. ``alfven_spdp_runtime.py --plot`` — the sp/dp panel across every node found

Steps 2 and 3 must be separate processes because ``jax_enable_x64`` is global
and has to be set before the first array is built -- which is exactly why this
driver exists rather than a single in-process loop.

Storage is automatic and node-derived. The GPU is picked **once** here via
autocvd and exported to every child, so (a) all series are measured on the same
device and are therefore comparable, and (b) each child files its results under
``pytests/mhd/data/astronomix/<node>/`` where ``<node>`` comes from the live
device (a100 / h100 / h200 / ...). Running this on new hardware creates a new
folder; it can never overwrite another GPU's numbers.

Options:
    --dry-run   print the plan (and the resolved GPU) without running anything
"""

# NOTE: autocvd must run before any child is spawned -- it is what sets
# CUDA_VISIBLE_DEVICES, which the children inherit instead of each grabbing
# (possibly different) GPUs of their own.
from autocvd import autocvd

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

DRY_RUN = "--dry-run" in sys.argv

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]

if not DRY_RUN:
    # One selection for the whole run; children see it via the environment.
    autocvd(num_gpus=1)

STEPS = [
    ("dp convergence sweep (FV JAX / FD JAX / FD Pallas)", [str(_HERE / "alfven_convergence.py")]),
    ("per-iteration runtime, double precision", [str(_HERE / "alfven_spdp_runtime.py"), "--dp"]),
    ("per-iteration runtime, single precision", [str(_HERE / "alfven_spdp_runtime.py"), "--sp"]),
    ("sp/dp panel (all nodes on disk)", [str(_HERE / "alfven_spdp_runtime.py"), "--plot"]),
]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO), env.get("PYTHONPATH", "")]))

    gpu = env.get("CUDA_VISIBLE_DEVICES", "<none>")
    print(f"[generate] GPU (via autocvd): CUDA_VISIBLE_DEVICES={gpu}")
    print(f"[generate] {len(STEPS)} steps; every child inherits this GPU\n")

    if DRY_RUN:
        for i, (title, cmd) in enumerate(STEPS, 1):
            print(f"  {i}. {title}\n     python -u {' '.join(cmd)}")
        print("\n[generate] --dry-run: nothing executed")
        return

    t_all = time.perf_counter()
    for i, (title, cmd) in enumerate(STEPS, 1):
        print(f"\n{'=' * 70}\n[generate] step {i}/{len(STEPS)}: {title}\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        # check=True: a failed sweep must not be papered over by later steps
        # appearing to succeed -- the data would silently be incomplete.
        subprocess.run([sys.executable, "-u", *cmd], check=True, env=env, cwd=str(_REPO))
        print(f"[generate] step {i} done in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\n[generate] ALL DONE in {(time.perf_counter() - t_all) / 60:.1f} min")
    print(f"[generate] data:    {_REPO / 'pytests' / 'mhd' / 'data' / 'astronomix'}/<node>/")
    print(f"[generate] figures: {_REPO / 'pytests' / 'mhd' / 'figures'}/<node>/")


if __name__ == "__main__":
    main()
