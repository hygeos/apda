"""Conversion module — exponential fit (Eq. 14) and inversion (Eq. 15).

Beer–Lambert predicts transmittance follows an exponential in PW. A
3-parameter empirical fit to LUT simulations:

    R_APDA ≈ exp(-(c + a · PW^b))                          [Eq. 14]

Inverted analytically for PW:

    PW = [(-ln R_APDA - c) / a]^(1/b)                      [Eq. 15]

Typical ranges for H₂O at 940 nm: a ∈ [0.12, 0.18], b ∈ [0.65, 0.80],
c ∈ [-0.05, 0.40].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from core import log


# ---------------------------------------------------------------------------
# Eq. 15 — ratio → PW inversion
# ---------------------------------------------------------------------------

def convert_ratio_to_pw(
    R: float | np.ndarray,
    a: float,
    b: float,
    c: float,
) -> float | np.ndarray:
    """Convert an APDA ratio to precipitable water column (Eq. 15).

    Inverts the empirical exponential model (Eq. 14):

        R ≈ exp(-(c + a · PW^b))
        ↓
        PW = [(-ln R - c) / a]^(1/b)

    Parameters
    ----------
    R : float or ndarray
        APDA ratio (dimensionless). Expected range (0, 1]:

        - R = 1.0  → no absorption  → PW = 0 when c = 0
        - R → 0    → full absorption → PW → +∞
        - exp(-c) < R ≤ 1 (only when c > 0) → below the model's PW = 0
          ratio (dry pixel / noise) → PW = 0
        - R > max(1, exp(-c)) → unphysical (noise / path-radiance error) → NaN
        - R ≤ 0    → undefined; R = 0 returns +∞, R < 0 returns NaN

    a : float
        Exponential scaling coefficient (typical H₂O 940 nm: 0.12 – 0.18).
    b : float
        Nonlinearity exponent (typical: 0.65 – 0.80).
    c : float
        Offset term (typical: −0.05 – 0.40).

    Returns
    -------
    float or ndarray
        Retrieved PW in the same units as the LUT (typically g/cm²).
        Scalar in → scalar out. Array in → array out.
    """
    R = np.asarray(R, dtype=float)
    scalar = R.ndim == 0
    orig_shape = R.shape

    # Work in 1-D flat space to avoid multi-dimensional fancy-indexing issues.
    R_flat = R.ravel()
    result = np.full(R_flat.shape, np.nan)

    # R = 0  →  ln(0) = -∞  →  PW = +∞ (complete absorption)
    result[R_flat == 0.0] = np.inf

    # R > 0  →  compute inner = (-ln R - c) / a
    pos = R_flat > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = (-np.log(R_flat[pos]) - c) / a

    # inner ≥ 0  →  valid physical PW
    valid = inner >= 0.0
    result[np.where(pos)[0][valid]] = inner[valid] ** (1.0 / b)

    # inner < 0  →  R > exp(-c): PW = 0 while R ≤ 1, unphysical (NaN) above
    dry = ~valid & (R_flat[pos] <= 1.0)
    result[np.where(pos)[0][dry]] = 0.0

    result = result.reshape(orig_shape)
    return float(result.ravel()[0]) if scalar else result


# ---------------------------------------------------------------------------
# Eq. 14 — fit a, b, c from LUT-derived ratios
# ---------------------------------------------------------------------------

def _exponential_model(pw: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Forward model: R = exp(-(c + a · PW^b)).  [Eq. 14]"""
    return np.exp(-(c + a * pw ** b))


def fit_exponential(
    pw_values: np.ndarray,
    ratios: np.ndarray,
    p0: tuple[float, float, float] = (0.15, 0.72, 0.10),
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
        = ((0.001, 0.1, -2.0), (5.0, 3.0, 5.0)),
    maxfev: int = 20_000,
) -> dict[str, float]:
    """Fit the exponential ratio-to-PW conversion parameters (Eq. 14).

    Fits ``R = exp(-(c + a · PW^b))`` by non-linear least squares to
    synthetic APDA ratios computed from the LUT.

    Parameters
    ----------
    pw_values : ndarray, shape (n_pw,)
        Water vapour column grid values from the LUT (g/cm²).
    ratios : ndarray, shape (n_pw,)
        Synthetic APDA ratios computed per PW node (dimensionless).
    p0 : tuple of 3 floats, default (0.15, 0.72, 0.10)
        Initial guess for (a, b, c).
    bounds : 2-tuple of 3-tuples
        Lower and upper bounds for (a, b, c).
    maxfev : int, default 20000
        Maximum function evaluations for ``curve_fit``.

    Returns
    -------
    dict[str, float]
        ``{"a": …, "b": …, "c": …}`` fitted parameters.

    Raises
    ------
    RuntimeError
        If fewer than 3 valid (finite, positive) ratio points, or if
        ``curve_fit`` fails to converge.
    """
    pw_values = np.asarray(pw_values, dtype=float)
    ratios = np.asarray(ratios, dtype=float)

    valid = np.isfinite(ratios) & (ratios > 0.0)
    if valid.sum() < 3:
        raise RuntimeError(
            f"Fewer than 3 valid synthetic APDA ratios ({valid.sum()} valid). "
            "Check LUT values and wavelength selections."
        )

    try:
        popt, _ = curve_fit(
            _exponential_model,
            pw_values[valid],
            ratios[valid],
            p0=list(p0),
            bounds=bounds,
            maxfev=maxfev,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Exponential fit to LUT ratios failed: {exc}") from exc

    a, b, c = popt
    log.debug(
        f"fit_exponential: a={a:.4f}  b={b:.4f}  c={c:.4f}  "
        f"({int(valid.sum())} PW points)"
    )
    return {"a": float(a), "b": float(b), "c": float(c)}
