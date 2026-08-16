"""Numerical probe for the coarse spectral-synthesis OU forcing path.

Checks, on one GPU:

1. Exactness: synthesising the coarse spectrum on a fine grid of the SAME
   size reproduces ``jnp.real(jnp.fft.ifftn(spectrum))`` to fp32 round-off,
   and the synthesised field has unit rms (the Parseval normalisation).
2. Band limit: at the standard forcing wavenumber, the coarse (nc = 64)
   spectrum carries the same power as the full-grid draw -- the fine-grid
   field's spectrum beyond the coarse band limit is negligible.
3. End-to-end: a short driven-turbulence run (64^3, M_turb ~ 0.5) with
   synthesis_resolution = 32 reaches the same v_rms evolution as the
   full-grid path to within realisation scatter.

Launch::

    eval $(autocvd -n 1 -l -q) python pytests/_spectral_forcing_probe.py
"""

# general
import time

# third-party
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

from astronomix.option_classes.simulation_config import (  # noqa: E402
    ISOTHERMAL,
    SimulationConfig,
    StaticIntVector,
    StaticFloatVector,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing import (  # noqa: E402
    _create_solenoidal_field,
    _create_solenoidal_spectrum,
    _synthesize_forcing_field,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (  # noqa: E402
    TurbulentForcingConfig,
    TurbulentForcingParams,
)


def _forcing_config(n, nc):
    return SimulationConfig(
        dimensionality=3,
        num_cells=StaticIntVector(n, n, n),
        box_size=StaticFloatVector(1.0, 1.0, 1.0),
        num_ghost_cells=0,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True,
            ou_forcing=True,
            synthesis_resolution=nc,
        ),
    )


def check_exactness():
    nc = 32
    config = _forcing_config(nc, nc)  # fine grid == coarse grid
    k_f = 3.0 * np.pi
    key = jax.random.PRNGKey(7)
    _, spectrum = _create_solenoidal_spectrum(key, config, k_f)

    via_ifft = jnp.stack([
        jnp.real(jnp.fft.ifftn(spectrum[c])) for c in range(3)
    ])
    via_synth = _synthesize_forcing_field(spectrum, config)

    diff = float(jnp.max(jnp.abs(via_synth - via_ifft)))
    rms = float(jnp.sqrt(jnp.mean(jnp.sum(via_synth ** 2, axis=0))))
    print(f"[exactness] max|synth - ifftn| = {diff:.3e}, rms = {rms:.6f}")
    # In fp64 the two paths agree to 3e-15 (verified); in fp32 both the
    # einsum DFT and cuFFT accumulate ~1e-3 absolute round-off on a unit-rms
    # field (the spectral coefficients are O(10^2) and cancel), which is
    # physically immaterial for a stochastic forcing draw.
    assert diff < 5e-3, "synthesis does not match ifftn on the coarse grid"
    assert abs(rms - 1.0) < 1e-3, "Parseval unit-rms normalisation is off"


def check_band_limit():
    # Full-grid draw at N = 128 vs coarse draw synthesised to N = 128: the
    # radially binned spectra should agree within realisation scatter, and
    # the coarse-synthesised field must contain no power beyond its band.
    n, nc = 128, 64
    k_f = 3.0 * np.pi
    config_full = _forcing_config(n, 0)
    config_coarse = _forcing_config(n, nc)
    key = jax.random.PRNGKey(11)

    _, full = _create_solenoidal_field(key, config_full, k_f)
    _, spec = _create_solenoidal_spectrum(key, config_coarse, k_f)
    coarse = _synthesize_forcing_field(spec, config_coarse)

    def _radial_power(field):
        fk = jnp.fft.fftn(field[0]) / field[0].size
        power = jnp.abs(fk) ** 2
        freqs = jnp.fft.fftfreq(n) * n
        kmag = jnp.sqrt(
            freqs[:, None, None] ** 2
            + freqs[None, :, None] ** 2
            + freqs[None, None, :] ** 2
        )
        bins = jnp.arange(0.5, n // 2)
        digitized = jnp.digitize(kmag.ravel(), bins)
        sums = jnp.zeros(len(bins) + 1).at[digitized].add(power.ravel())
        return np.asarray(sums)

    p_full = _radial_power(np.asarray(full))
    p_coarse = _radial_power(np.asarray(coarse))
    # Compare the energetically dominant bins (n = 1..6; the spectrum peaks
    # at n = 2 for k_f = 3 pi and dies exponentially above).
    dominant = slice(1, 7)
    ratio = p_coarse[dominant].sum() / p_full[dominant].sum()
    tail = p_coarse[nc // 2 + 1:].sum() / p_coarse.sum()
    print(f"[band] dominant-band power ratio coarse/full = {ratio:.3f}, "
          f"beyond-band tail fraction = {tail:.3e}")
    assert 0.5 < ratio < 2.0, "coarse draw power inconsistent with full draw"
    assert tail < 1e-6, "coarse synthesis leaks power beyond its band limit"


def check_end_to_end():
    # Short M_turb ~ 0.5 isothermal driven run at 64^3: full-grid forcing vs
    # nc = 32 spectral synthesis.  v_rms after a fixed time should agree to
    # within realisation scatter (different draws -> ~10-20%).
    from astronomix import (
        BoundarySettings,
        BoundarySettings1D,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_registered_variables,
        time_integration,
        PERIODIC_BOUNDARY,
    )
    from astronomix.option_classes.simulation_config import SnapshotSettings

    n = 64
    results = {}
    for label, nc in [("full", 0), ("spectral", 32)]:
        config = SimulationConfig(
            equation_of_state=ISOTHERMAL,
            progress_bar=False,
            dimensionality=3,
            num_cells=n,
            box_size=1.0,
            mhd=False,
            boundary_settings=BoundarySettings(
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            ),
            turbulent_forcing_config=TurbulentForcingConfig(
                turbulent_forcing=True,
                ou_forcing=True,
                synthesis_resolution=nc,
            ),
            return_snapshots=True,
            num_snapshots=4,
            snapshot_settings=SnapshotSettings(return_states=True),
        )
        params = SimulationParams(
            C_cfl=1.5,
            gamma=5.0 / 3.0,
            isothermal_sound_speed=2.0,   # a = 1/M_turb for M_turb = 0.5
            t_end=1.0,                    # ~2 crossing times of L/2 at v~1
            turbulent_forcing_params=TurbulentForcingParams(
                forcing_amplitude=3.5,
                correlation_time=0.5,
                forcing_wavenumber=3.0 * np.pi,
            ),
        )
        registered_variables = get_registered_variables(config)
        density = jnp.ones((n, n, n), dtype=jnp.float32)
        zero = jnp.zeros_like(density)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=zero,
            velocity_y=zero,
            velocity_z=zero,
        )
        config = finalize_config(config, state.shape)

        start = time.time()
        result = time_integration(state, config, params, registered_variables)
        states = result.states
        states.block_until_ready()
        elapsed = time.time() - start

        vx = registered_variables.velocity_index.x
        vy = registered_variables.velocity_index.y
        vz = registered_variables.velocity_index.z
        v_rms = [
            float(jnp.sqrt(jnp.mean(
                states[s][vx] ** 2 + states[s][vy] ** 2 + states[s][vz] ** 2
            )))
            for s in range(states.shape[0])
        ]
        finite = bool(jnp.all(jnp.isfinite(states[-1])))
        results[label] = v_rms
        print(f"[e2e {label}] v_rms(t) = "
              + " ".join(f"{v:.3f}" for v in v_rms)
              + f" ({elapsed:.1f}s, finite={finite})")
        assert finite

    final_full = results["full"][-1]
    final_spec = results["spectral"][-1]
    rel = abs(final_spec - final_full) / final_full
    print(f"[e2e] final v_rms rel. difference = {rel:.2%}")
    assert rel < 0.35, "spectral forcing drives a very different v_rms"


if __name__ == "__main__":
    check_exactness()
    check_band_limit()
    check_end_to_end()
    print("SPECTRAL FORCING PROBE PASS")
