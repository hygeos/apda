"""Path radiance module — empirical L_atm adjustment (Eq. 17).

Because the initial aerosol assumptions carry uncertainty, the ATBD
specifies an empirical scaling of the path radiance term:

    L_atm,i = L_atm,i^M · (1 + α · g_i / g_max)             [Eq. 17]

where:

- ``L_atm,i^M``: initial MODTRAN-derived path radiance estimate.
- ``α``: empirical adjustment parameter (optimised per scene; ATBD
  reports -0.10 to -0.18 for the test scenes).
- ``g_i``: per-channel weighting factor — the dynamic range of
  ``L_atm,i`` across atmospheric states (PW nodes) in the LUT.
- ``g_max``: maximum weighting factor across all channels.

The parameter ``α`` is found by minimising the standard deviation of
retrieved PW over a spectrally diverse, flat image subset:

    σ(PW_subset) → MINIMUM                                 [Eq. 16]
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core import log


# ---------------------------------------------------------------------------
# g_i weights
# ---------------------------------------------------------------------------

def compute_g_weights(
    rho_atm: xr.DataArray,
    pw_dim: str = "WV",
) -> xr.DataArray:
    """Compute per-channel dynamic-range weights g_i (Eq. 17).

    The weight ``g_i`` measures how strongly the path radiance in channel
    ``i`` varies across the PW range of the LUT. Channels with high
    dynamic range are more sensitive to aerosol/state uncertainty and
    receive a higher weight in the α adjustment.

    Parameters
    ----------
    rho_atm : xr.DataArray
        Atmospheric path reflectance, with a PW dimension (``WV`` or
        ``pw``). Shape: ``(WV, wavelength)`` or with extra geometry dims.
    pw_dim : str, default "WV"
        Name of the PW dimension in ``rho_atm``.

    Returns
    -------
    xr.DataArray
        Per-channel weights ``g_i``, normalised so that ``g_max = 1``.
        Dims: ``wavelength`` (geometry dims, if any, are averaged out).
    """
    if pw_dim not in rho_atm.dims:
        raise ValueError(
            f"PW dimension '{pw_dim}' not found in rho_atm dims: {rho_atm.dims}"
        )

    # Dynamic range across PW nodes: max - min per channel.
    # Collapse any non-PW, non-wavelength dims (e.g. rho) by averaging.
    other_dims = tuple(d for d in rho_atm.dims if d not in (pw_dim, "wavelength"))
    if other_dims:
        rho_atm = rho_atm.mean(dim=other_dims)

    g = rho_atm.max(dim=pw_dim) - rho_atm.min(dim=pw_dim)
    g_max = float(g.max())

    if g_max <= 0:
        log.warning(
            "compute_g_weights: g_max <= 0 (rho_atm flat across PW). "
            "Weights set to 0 → α adjustment will have no effect."
        )
        return xr.zeros_like(g)

    g_norm = g / g_max
    g_norm.attrs["units"] = "dimensionless"
    g_norm.attrs["description"] = "Per-channel path-radiance dynamic-range weight (g_i/g_max)"
    log.debug(f"compute_g_weights: g_max={g_max:.4g}, n_channels={g.size}")
    return g_norm


# ---------------------------------------------------------------------------
# Eq. 17 — scale path radiance
# ---------------------------------------------------------------------------

def scale_path_radiance(
    rho_atm: xr.DataArray,
    alpha: float,
    g_weights: xr.DataArray,
) -> xr.DataArray:
    """Apply the empirical path radiance adjustment (Eq. 17).

        L_atm,i = L_atm,i^M · (1 + α · g_i / g_max)

    Parameters
    ----------
    rho_atm : xr.DataArray
        Initial path reflectance from the LUT (``L_atm,i^M``).
    alpha : float
        Empirical adjustment parameter. ATBD typical range: [-0.18, -0.10].
    g_weights : xr.DataArray
        Normalised per-channel weights from :func:`compute_g_weights`
        (already divided by ``g_max``).

    Returns
    -------
    xr.DataArray
        Adjusted path reflectance, same shape as ``rho_atm``.
    """
    # Broadcast g_weights (wavelength,) onto rho_atm.
    scaled = rho_atm * (1.0 + alpha * g_weights)
    scaled.attrs = dict(rho_atm.attrs)
    scaled.attrs["alpha"] = alpha
    scaled.attrs["description"] = (
        f"Path reflectance adjusted by Eq. 17 (alpha={alpha:.4f})"
    )
    log.debug(f"scale_path_radiance: alpha={alpha:.4f}")
    return scaled
