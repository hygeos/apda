"""apda vs the CORATHYP reference (REF) on the shared numerical core.

The LIR continuum and the APDA ratio must agree to rounding: apda's ``"ols"``
continuum is REF's ``calc_lir(method="linear")`` (``polyfit`` deg 1), and
``"spline"`` matches REF's natural cubic spline. Uses synthetic data only;
the comparison on the real LUT / HYPS dataset is ``validation.compare_ref``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from apda import ChannelSet, compute_apda_ratio
from apda.ratio import interp_continuum
from validation.compare_ref import ref_instance, ref_invert, ref_lut_bands, ref_ratio_curve

WL_R = np.array([0.85, 0.86, 0.87, 0.88, 0.98, 0.99, 1.00])
LAMBDA_M = 0.91


def ref_lir(wl_r, values, lambda_m, method="linear"):
    """REF ``calc_lir`` on (..., n_ref) values, with ``lambda_m`` preset."""
    ref = ref_instance(ChannelSet("t", meas=(lambda_m,), ref=tuple(wl_r)))
    ref._lambda_mean_abs_band = lambda_m
    ref.calc_lambda_mean_in_absorption = lambda: None  # needs a full LUT otherwise
    values = np.asarray(values, dtype=float)
    dims = ["x", "y", "lambda_eq"][3 - values.ndim:]
    return ref.calc_lir(xr.DataArray(values, dims=dims, coords={"lambda_eq": wl_r}), method).values


class TestLIR:
    def test_ols_matches_ref_1d(self):
        v = np.array([0.40, 0.42, 0.41, 0.43, 0.55, 0.54, 0.56])
        assert interp_continuum(WL_R, v, LAMBDA_M, "ols") == pytest.approx(ref_lir(WL_R, v, LAMBDA_M), abs=1e-12)

    def test_ols_matches_ref_random_image(self):
        v = np.random.default_rng(5).random((4, 3, WL_R.size)) * 0.4 + 0.3
        np.testing.assert_allclose(interp_continuum(WL_R, v, LAMBDA_M, "ols"),
                                   ref_lir(WL_R, v, LAMBDA_M), rtol=1e-12)

    def test_spline_matches_ref(self):
        v = np.random.default_rng(3).random((2, 2, WL_R.size)) * 0.4 + 0.3
        np.testing.assert_allclose(interp_continuum(WL_R, v, LAMBDA_M, "spline"),
                                   ref_lir(WL_R, v, LAMBDA_M, "spline"), rtol=1e-12)

    def test_ols_differs_from_piecewise(self):
        """With > 2 ref channels apda "linear" is not REF's "linear"."""
        v = np.array([0.40, 0.60, 0.41, 0.43, 0.55, 0.54, 0.56])
        assert abs(interp_continuum(WL_R, v, LAMBDA_M, "ols")
                   - interp_continuum(WL_R, v, LAMBDA_M, "linear")) > 1e-6


def test_ratio_curve_matches_ref():
    """R(WV) on a toy LUT: apda compute_apda_ratio vs REF calc_apda_on_lut."""
    ch = ChannelSet("t", meas=(0.90, 0.91, 0.92), ref=(0.86, 0.87, 0.98, 0.99))
    wl = np.array(sorted(ch.all))
    wv = np.array([0.5, 1.0, 2.0, 4.0])
    rho = np.array([0.0, 0.5])
    rng = np.random.default_rng(0)
    rho_toa = xr.DataArray(
        rng.random((wv.size, rho.size, wl.size)) * 0.3 + 0.1 * rho[:, None],
        dims=("WV", "rho", "wavelength"), coords={"WV": wv, "rho": rho, "wavelength": wl},
    )
    lb = ref_lut_bands(xr.Dataset({"rho_toa": rho_toa}), ch)
    r_ref = ref_ratio_curve(ref_instance(ch), lb).sel(rho=0.5).values

    corr = (lb.sel(rho=0.5) - lb.sel(rho=0)).values
    r_apda = compute_apda_ratio(corr[:, :3], corr[:, 3:], np.array(ch.ref), float(np.mean(ch.meas)))
    np.testing.assert_allclose(r_apda, r_ref, rtol=1e-12)


def test_ref_inversion_round_trip():
    alpha, beta = 0.05, 0.3
    pw = np.array([0.3, 1.0, 4.0])
    np.testing.assert_allclose(ref_invert(np.exp(-(alpha + beta * np.sqrt(pw))), alpha, beta), pw)
