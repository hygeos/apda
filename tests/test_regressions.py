"""Regression tests for issues fixed after the v0.1 audit (all failed on ``b5bdff5``).

1. NaN geometry was snapped to the first LUT node → now gives NaN.
2. With a fitted ``c > 0``, ratios in ``(exp(-c), 1]`` (dry / noisy pixels)
   gave NaN → now PW = 0.
3. A failed refit kept the previous model; ``n_iter=0`` crashed with an
   AttributeError → params cleared before fitting; clear ValueError.
4. Out-of-range geometry and PW were clamped silently → clamped with a warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from apda import APDALUT, convert_ratio_to_pw, retrieve
from tests.test_apda import A, B, C, CHANNELS, RHO_SURF, fitted_retriever, lut_path, toy_scene, toy_toa  # noqa: F401

# 1. NaN geometry ------------------------------------------------------------

def test_nan_geometry_gives_nan(lut_path):
    """A pixel with NaN geometry must come out NaN, not snapped to LUT node 0."""
    lut = APDALUT(lut_path())
    sza = xr.DataArray([[np.nan, 44.0]], dims=("y", "x"))
    pw = retrieve(lut, toy_scene([[1.0, 3.0]], sza=sza), CHANNELS, rho=RHO_SURF)
    assert np.isnan(pw.values[0, 0])
    assert pw.values[0, 1] == pytest.approx(3.0, rel=1e-3)


# 2. Inversion domain ----------------------------------------------------------

def test_ratio_between_exp_minus_c_and_one_is_finite():
    """With c > 0, R in (exp(-c), 1] (dry / noisy pixel) must not be NaN."""
    assert C > 0
    R = np.array([np.exp(-C) + 1e-3, 0.99, 1.0])
    pw = convert_ratio_to_pw(R, A, B, C)
    assert np.all(np.isfinite(pw)) and np.all(pw >= 0)


def test_dry_pixel_retrieval_is_finite(lut_path):
    """A slightly brighter-than-model measurement band must not NaN the pixel."""
    retriever = fitted_retriever(lut_path)
    toa = toy_toa([[0.5]])
    toa[..., 1:3] = toa[..., 0]  # no absorption at all: R ≈ 1 > exp(-c)
    with pytest.warns(UserWarning, match="'WV'"):  # PW = 0 is below the LUT's first WV node
        pw = retriever.retrieve(toa)
    assert (pw.values == 0).all()


# 3. Stale fit state / n_iter=0 --------------------------------------------------

def test_failed_refit_clears_params(lut_path):
    """If a refit raises, retrieve() must not silently reuse the previous fit."""
    retriever = fitted_retriever(lut_path)
    retriever._lut = retriever._lut.assign(rho_toa=retriever._lut["rho_toa"] * np.nan)  # unfittable
    with pytest.raises(Exception):
        retriever.fit()
    with pytest.raises(RuntimeError):
        retriever.retrieve(toy_toa([[2.0]]))


def test_n_iter_zero(lut_path):
    """n_iter=0 must raise a clear ValueError, not an AttributeError."""
    retriever = fitted_retriever(lut_path)
    with pytest.raises(ValueError, match="n_iter"):
        retriever.retrieve(toy_toa([[2.0]]), pw_initial=2.0, n_iter=0)


# 4. Silent clamping -----------------------------------------------------------

def test_out_of_range_geometry_warns(lut_path):
    """sza=0 (LUT starts at 22°) must warn, not clamp silently."""
    lut = APDALUT(lut_path())
    with pytest.warns(UserWarning):
        lut.select_geometry(sza=0.0, vza=0.0, raa=90.0, aot=0.15)


def test_in_range_geometry_does_not_warn(lut_path):
    """Guard against over-warning once the fix lands."""
    lut = APDALUT(lut_path())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lut.select_geometry(sza=44.0, vza=0.0, raa=90.0, aot=0.15)


def test_out_of_range_pw_warns(lut_path):
    """PW outside the LUT WV range must warn, not clamp silently."""
    retriever = fitted_retriever(lut_path)
    with pytest.warns(UserWarning):
        retriever.retrieve(toy_toa([[2.0]]), pw_initial=20.0, n_iter=1)
