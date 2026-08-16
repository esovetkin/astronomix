"""
Render the turbulent power spectra from a ``turb_diagnostics.py`` NPZ.

Three panels: kinetic energy spectrum at the final epoch (plus its
compensated form to expose the inertial range), magnetic energy spectra
across epochs (the dynamo growth), and the density spectrum at the final
epoch.  Wavenumbers are in units of the box mode k_min = 2 pi / L, i.e.
the mode number n; the OU forcing peaks at n = 2.  A least-squares slope
over the inertial range is printed for the kinetic spectrum.

    python examples/scripts/forward/mhd/turbulence/plot_spectra.py \
        --diag diag_icm2048.npz --tcross 0.5 --out spectra.png
"""

# general
import argparse

# third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--diag", type=str, required=True, help="NPZ from turb_diagnostics.py")
parser.add_argument("--tcross", type=float, default=0.5)
parser.add_argument("--fit-range", type=str, default="5,60",
                    help="mode-number range for the kinetic slope fit")
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

data = np.load(args.diag)
steps_with_spectra = sorted(
    int(key.split("_")[-1]) for key in data.keys() if key.startswith("spec_EK_")
)
step_list = list(data["steps"])
t_over_tc = {s: float(data["t_over_tc"][step_list.index(s)]) for s in steps_with_spectra}
final = steps_with_spectra[-1]

ink = "#333333"
plt.rcParams.update({
    "text.color": ink, "axes.labelcolor": ink,
    "xtick.color": ink, "ytick.color": ink,
})

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

# --------------------------------------------------------------------------
# Panel 1: kinetic spectrum at the final epoch + inertial-range slope fit.
# --------------------------------------------------------------------------
ax = axes[0]
spectrum = np.asarray(data[f"spec_EK_{final}"])
modes = np.arange(len(spectrum))
valid = modes >= 1
ax.loglog(modes[valid], spectrum[valid], color="#2c6fbb", lw=1.8,
          label=f"$E_K(k)$, $t/t_c$ = {t_over_tc[final]:.1f}")

lo, hi = (int(x) for x in args.fit_range.split(","))
fit_mask = (modes >= lo) & (modes <= hi) & (spectrum > 0)
slope, intercept = np.polyfit(
    np.log(modes[fit_mask]), np.log(spectrum[fit_mask]), 1
)
print(f"kinetic inertial-range slope (n = {lo}..{hi}): {slope:.2f}")

anchor = np.exp(intercept)
guide = modes[fit_mask].astype(float)
ax.loglog(guide, 1.6 * anchor * guide ** (-5.0 / 3.0), "--", color="#999999",
          lw=1.2, label=r"$k^{-5/3}$")
ax.axvline(2, color="#cccccc", lw=1, zorder=0)
ax.text(2.1, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e-12,
        "forcing", fontsize=9, color="#888888", rotation=90, va="bottom")
ax.set_title(f"kinetic spectrum (fit $n$={lo}..{hi}: "
             f"slope = {slope:.2f})", fontsize=12)

# --------------------------------------------------------------------------
# Panel 2: magnetic spectra across epochs (dynamo growth), colored by time.
# --------------------------------------------------------------------------
ax = axes[1]
colormap = plt.get_cmap("viridis")
for i, s in enumerate(steps_with_spectra):
    spectrum = np.asarray(data[f"spec_EB_{s}"])
    modes = np.arange(len(spectrum))
    valid = (modes >= 1) & (spectrum > 0)
    ax.loglog(modes[valid], spectrum[valid],
              color=colormap(i / max(1, len(steps_with_spectra) - 1)), lw=1.3)
guide_n = np.array([2.0, 30.0])
eb_first = np.asarray(data[f"spec_EB_{steps_with_spectra[0]}"])
ax.loglog(guide_n, eb_first[2] * 0.5 * (guide_n / 2.0) ** 1.5, "--",
          color="#999999", lw=1.2, label=r"Kazantsev $k^{3/2}$")
scalar_map = plt.cm.ScalarMappable(
    cmap=colormap,
    norm=plt.Normalize(t_over_tc[steps_with_spectra[0]], t_over_tc[final]),
)
colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.02)
colorbar.set_label(r"$t / t_\mathrm{cross}$")
colorbar.outline.set_edgecolor("#bbbbbb")
ax.set_title("magnetic spectra (dynamo growth)", fontsize=12)

# --------------------------------------------------------------------------
# Panel 3: density spectrum at the final epoch.
# --------------------------------------------------------------------------
ax = axes[2]
spectrum = np.asarray(data[f"spec_rho_{final}"])
modes = np.arange(len(spectrum))
valid = (modes >= 1) & (spectrum > 0)
ax.loglog(modes[valid], spectrum[valid], color="#b05a2c", lw=1.8,
          label=f"$P_\\rho(k)$, $t/t_c$ = {t_over_tc[final]:.1f}")
guide = np.geomspace(5, 60, 20)
ax.loglog(guide, spectrum[5] * 1.6 * (guide / 5.0) ** (-5.0 / 3.0), "--",
          color="#999999", lw=1.2, label=r"$k^{-5/3}$")
ax.set_title("density spectrum", fontsize=12)

for ax in axes:
    ax.set_xlabel(r"$k / k_\mathrm{min}$ (mode number $n$)")
    ax.set_ylabel("shell-integrated power")
    ax.legend(frameon=False, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
    ax.tick_params(length=3)

fig.suptitle(
    f"Turbulent power spectra, $2048^3$, $M_\\mathrm{{turb}} \\approx 0.44$, "
    f"$\\beta_0 = 10^6$",
    fontsize=14,
)
fig.savefig(args.out, dpi=140)
print(f"wrote {args.out}")
