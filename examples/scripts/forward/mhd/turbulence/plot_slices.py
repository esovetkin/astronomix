"""
Render the mid-plane slices extracted by ``turb_slices.py``.

Produces a 2x2 panel figure: density (top row) and magnetic energy
(bottom row), each in the x-y plane at z = L/2 and the y-z plane at
x = L/2.  Density is linear (the run is subsonic, contrast is a few
percent around the mean); magnetic energy spans decades and gets a log
color scale with robust (0.5 / 99.9 percentile) limits.  Both use
perceptually uniform sequential colormaps (magnitude data -> one ramp,
light-to-dark; never a rainbow).

    python examples/scripts/forward/mhd/turbulence/plot_slices.py \
        --slices slices_icm2048_final.npz --tcross 0.5 --out slices.png
"""

# general
import argparse

# third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument("--slices", type=str, required=True, help="NPZ from turb_slices.py")
parser.add_argument("--tcross", type=float, default=0.5, help="crossing time (for the title)")
parser.add_argument("--out", type=str, required=True, help="output PNG path")
args = parser.parse_args()

data = np.load(args.slices)
time = float(data["time"])
step = int(data["step"])

# Robust color limits: density symmetric around the mean; magnetic energy
# log-scaled between its 0.5th and 99.9th percentile so single extreme
# cells do not crush the ramp.
rho_all = np.concatenate([data["rho_xy"].ravel(), data["rho_yz"].ravel()])
eb_all = np.concatenate([data["EB_xy"].ravel(), data["EB_yz"].ravel()])
rho_limits = (np.percentile(rho_all, 0.1), np.percentile(rho_all, 99.9))
eb_limits = (np.percentile(eb_all, 0.5), np.percentile(eb_all, 99.9))

panels = [
    ("rho_xy", r"density $\rho$", "x", "y", dict(cmap="viridis",
        vmin=rho_limits[0], vmax=rho_limits[1])),
    ("rho_yz", r"density $\rho$", "y", "z", dict(cmap="viridis",
        vmin=rho_limits[0], vmax=rho_limits[1])),
    ("EB_xy", r"magnetic energy $B^2/2$", "x", "y", dict(cmap="magma",
        norm=LogNorm(vmin=eb_limits[0], vmax=eb_limits[1]))),
    ("EB_yz", r"magnetic energy $B^2/2$", "y", "z", dict(cmap="magma",
        norm=LogNorm(vmin=eb_limits[0], vmax=eb_limits[1]))),
]

ink = "#333333"
plt.rcParams.update({
    "text.color": ink, "axes.labelcolor": ink,
    "xtick.color": ink, "ytick.color": ink,
})

fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5), constrained_layout=True)
for ax, (key, label, xlab, ylab, imshow_kwargs) in zip(axes.ravel(), panels):
    # Arrays are indexed [first_axis, second_axis]; imshow wants the first
    # axis vertical, so transpose and set origin so xlab runs rightward.
    image = ax.imshow(
        data[key].T, origin="lower", extent=(0, 1, 0, 1),
        interpolation="nearest", rasterized=True, **imshow_kwargs,
    )
    plane = f"{xlab}-{ylab} plane at {('z' if 'xy' in key else 'x')} = L/2"
    ax.set_title(f"{label}, {plane}", fontsize=12)
    ax.set_xlabel(f"{xlab} / L")
    ax.set_ylabel(f"{ylab} / L")
    ax.tick_params(length=3)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    colorbar.outline.set_edgecolor("#bbbbbb")

fig.suptitle(
    f"Driven MHD turbulence, $2048^3$, $M_\\mathrm{{turb}} \\approx 0.44$, "
    f"$\\beta_0 = 10^6$ — checkpoint {step}, "
    f"$t/t_\\mathrm{{cross}} = {time / args.tcross:.1f}$",
    fontsize=14,
)
fig.savefig(args.out, dpi=140)
print(f"wrote {args.out}")
