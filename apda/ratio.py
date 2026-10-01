"""APDA ratio (Eq. 6) and reference-continuum interpolation (LIR).

Unit-agnostic: works identically on radiances or reflectances, as long as
inputs are consistent — any common scale factor (e.g. E₀/π) cancels.

    R_APDA = mean_i [L_m,i - L_atm,m,i]
             ─────────────────────────────────────────────────
             LIR([λ_r,j], [L_r,j - L_atm,r,j]) evaluated at λ_m

where LIR is the regression line through the corrected reference
continuum, evaluated at the mean measurement wavelength λ_m
(Schläpfer et al. 1998, Eq. 6).

Both functions are vectorised: the spectral axis is the last one, any
leading axes are pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

#: Reference-continuum methods.
LIR_METHODS: tuple[str, ...] = ("ols", "linear", "spline")


def interp_continuum(
    wavelengths_r: np.ndarray,
    values_r_corr: np.ndarray,
    lambda_m: float,
    method: str = "ols",
) -> np.ndarray:
    """Evaluate the reference continuum at ``lambda_m``.

    Parameters
    ----------
    wavelengths_r : ndarray, shape (n_ref,)
        Reference channel wavelengths, in any order.
    values_r_corr : ndarray, shape (..., n_ref)
        Path-corrected reference values.
    lambda_m : float
        Evaluation wavelength (mean measurement wavelength).
    method : {"ols", "linear", "spline"}, default "ols"
        - ``"ols"``: least-squares line through all reference points (the
          ATBD Eq. 6 definition); extrapolates.
        - ``"linear"``: piecewise linear between the bracketing channels.
        - ``"spline"``: natural cubic spline (linear below 3 channels).

        ``"linear"`` and ``"spline"`` clamp ``lambda_m`` to the reference
        range.

    Returns
    -------
    ndarray, shape (...)
        Continuum value(s) at ``lambda_m``.
    """
    if method not in LIR_METHODS:
        raise ValueError(f"Unknown LIR method '{method}'. Use one of {LIR_METHODS}.")

    wavelengths_r = np.asarray(wavelengths_r, dtype=float)
    order = np.argsort(wavelengths_r)
    x = wavelengths_r[order]
    y = np.asarray(values_r_corr, dtype=float)[..., order]

    if x.size == 1:
        return y[..., 0]

    if method == "ols":
        dx = x - x.mean()
        y_mean = y.mean(axis=-1)
        slope = np.sum(dx * (y - y_mean[..., None]), axis=-1) / np.sum(dx**2)
        return y_mean + slope * (lambda_m - x.mean())

    xm = float(np.clip(lambda_m, x[0], x[-1]))
    if method == "spline" and x.size >= 3:
        return CubicSpline(x, y, axis=-1, bc_type="natural")(xm)

    i = int(np.clip(np.searchsorted(x, xm), 1, x.size - 1))
    w = (xm - x[i - 1]) / (x[i] - x[i - 1])
    return y[..., i - 1] + w * (y[..., i] - y[..., i - 1])


def compute_apda_ratio(
    values_m_corr: np.ndarray,
    values_r_corr: np.ndarray,
    wavelengths_r: np.ndarray,
    lambda_m: float,
    interp_method: str = "ols",
) -> np.ndarray:
    """APDA ratio (Eq. 6) from path-corrected measurement/reference values.

    Parameters
    ----------
    values_m_corr : ndarray, shape (..., n_meas)
        Corrected measurement-channel values.
    values_r_corr : ndarray, shape (..., n_ref)
        Corrected reference-channel values.
    wavelengths_r : ndarray, shape (n_ref,)
        Reference channel wavelengths.
    lambda_m : float
        Mean measurement wavelength.
    interp_method : {"ols", "linear", "spline"}, default "ols"
        Continuum method (see :func:`interp_continuum`).

    Returns
    -------
    ndarray, shape (...)
        Dimensionless ratio; NaN where the continuum is zero (dark target).
    """
    numerator = np.mean(values_m_corr, axis=-1)
    denominator = interp_continuum(wavelengths_r, values_r_corr, lambda_m, interp_method)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator != 0.0, numerator / denominator, np.nan)
    return ratio[()]
