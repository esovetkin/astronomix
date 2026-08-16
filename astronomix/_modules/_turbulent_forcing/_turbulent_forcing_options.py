"""
Configuration and parameter containers for turbulent forcing.

``TurbulentForcingConfig`` holds the static switches (forcing on/off, vacuum
protection, and the choice of Ornstein-Uhlenbeck versus white-in-time forcing),
while ``TurbulentForcingParams`` holds the tunable physical parameters.
"""

# typing
from typing import NamedTuple


class TurbulentForcingConfig(NamedTuple):
    vacuum_protection: bool = False
    turbulent_forcing: bool = False

    #: Use Ornstein-Uhlenbeck (temporally correlated) forcing instead of the
    #: default white-in-time forcing. The OU field persists across steps and is
    #: evolved as f <- a f + sqrt(1 - a^2) xi (a = exp(-dt / correlation_time)),
    #: with xi a fresh unit-rms solenoidal field peaked at forcing_wavenumber.
    #: It is applied as a constant-amplitude acceleration (velocity += F0 f dt),
    #: which is state-independent (clean adjoint) and -- unlike the white
    #: forcing -- lets rotation organise coherent structures (columns).
    ou_forcing: bool = False

    #: Coarse spectral-synthesis resolution for the OU forcing. ``0`` (the
    #: default) keeps the original full-grid path: each fresh solenoidal draw
    #: is synthesised by an inverse FFT on the whole simulation grid.  A
    #: positive value ``nc`` instead draws and OU-evolves the forcing in
    #: spectral space on a small ``nc^3`` grid and evaluates the band-limited
    #: field on the fine grid by exact per-axis inverse-DFT matrix products.
    #: Because the forcing spectrum k^6 exp(-8 k / kpk) is negligible beyond a
    #: few kpk, ``nc = 64`` already carries the full spectrum to fp32
    #: precision for the standard k_f -- while making the forcing scale to
    #: sharded multi-node grids (the full-grid inverse FFT would otherwise be
    #: replicated per device by the SPMD partitioner) and cutting its cost at
    #: large N.  The OU update is linear, so evolving the spectrum and
    #: synthesising afterwards is mathematically identical to evolving the
    #: real-space field.
    synthesis_resolution: int = 0


class TurbulentForcingParams(NamedTuple):
    protection_density_threshold: float = 0.02
    protection_max_velocity: float = 50.0
    energy_injection_rate: float = 2.0

    #: OU forcing correlation time tau_f (~ one eddy turnover). Only used when
    #: TurbulentForcingConfig.ou_forcing is True.
    correlation_time: float = 1.0

    #: OU forcing peak wavenumber k_f (in physical units, k = 2 pi n / L). The
    #: solenoidal forcing spectrum k^6 exp(-8 k / kpk) is peaked at k_f by
    #: setting kpk = k_f / 0.75. Only used when ou_forcing is True.
    forcing_wavenumber: float = 4.0

    #: OU forcing amplitude F0 (acceleration scale); tunes the stationary
    #: u_rms. Only used when ou_forcing is True.
    forcing_amplitude: float = 1.0
