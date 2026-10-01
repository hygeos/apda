"""Unit tests for apda, on a synthetic operational-format LUT.

The toy LUT follows ``rho_toa = rho · T_wv + rho_atm`` with
``T_wv = exp(-(c + a·PW^b))`` on the measurement channels and 1 on the
reference channels, so the retrieval must recover a, b, c and PW exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from apda import (
    APDA,
    APDALUT,
    ChannelSet,
    compute_apda_ratio,
    convert_ratio_to_pw,
    fit_exponential,
    retrieve_scene,
    select_channels,
)
from apda.bands import CURATED_BANDS
from apda.path_radiance import compute_g_weights, scale_path_radiance
from apda.ratio import interp_continuum

A, B, C = 0.15, 0.72, 0.10
PW_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
CHANNELS = ChannelSet("toy", meas=(0.91, 0.92), ref=(0.87, 0.96))
WL = (0.87, 0.91, 0.92, 0.96)
RHO_SURF = 0.5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_toy_lut(rho_atm=0.05) -> xr.Dataset:
    """Toy LUT; ``rho_atm`` is a scalar or a per-PW profile."""
    pw = np.array(PW_GRID)
    wl = np.array(WL)
    is_meas = np.isin(wl, CHANNELS.meas)
    t_wv = np.where(is_meas, np.exp(-(C + A * pw[:, None] ** B)), 1.0)  # (WV, wl)
    atm = np.broadcast_to(np.asarray(rho_atm, dtype=float).reshape(-1, 1), t_wv.shape)

    coords = {
        "aot": [0.05, 0.15, 0.4], "sza": [22.0, 44.0, 76.0], "vza": [0.0, 30.0],
        "raa": [0.0, 90.0, 180.0], "WV": pw, "rho": [0.0, 0.3, 0.5, 1.0], "wavelength": wl,
    }
    rho_atm_da = xr.DataArray(atm, dims=("WV", "wavelength"))
    rho_toa_da = xr.DataArray(coords["rho"], dims="rho") * xr.DataArray(t_wv, dims=("WV", "wavelength")) + rho_atm_da
    geom = xr.DataArray(
        np.ones((3, 3, 2, 3)), dims=("aot", "sza", "vza", "raa")
    )
    ds = xr.Dataset({"rho_atm": rho_atm_da * geom, "rho_toa": rho_toa_da * geom}, coords=coords)
    return ds.transpose("aot", "sza", "WV", "rho", "vza", "raa", "wavelength")


@pytest.fixture
def lut_path(tmp_path):
    def _write(**kwargs):
        path = tmp_path / "toy_lut.nc"
        make_toy_lut(**kwargs).to_netcdf(path)
        return path
    return _write


def fitted_retriever(lut_path, interp_method="ols", channels=CHANNELS, **lut_kwargs):
    lut = APDALUT(lut_path(**lut_kwargs))
    lut.validate()
    collapsed = lut.collapse(44.0, 0.0, 90.0, 0.15, channels, rho=RHO_SURF)
    retriever = APDA(collapsed, channels, interp_method=interp_method)
    retriever.fit()
    return retriever


def toy_toa(pw_true, rho_atm=0.05, rho_surf=RHO_SURF) -> xr.DataArray:
    """(y, x, wavelength) image for a 2-D grid of true PW values."""
    pw_true = np.asarray(pw_true, dtype=float)
    t = np.exp(-(C + A * pw_true ** B))
    meas = rho_surf * t + rho_atm
    ref = np.broadcast_to(rho_surf + rho_atm, meas.shape)
    return xr.DataArray(
        np.stack([ref, meas, meas, ref], axis=-1),
        dims=("y", "x", "wavelength"), coords={"wavelength": list(WL)},
    )


# ---------------------------------------------------------------------------
# ratio.py
# ---------------------------------------------------------------------------

WL_7 = np.array([0.85, 0.86, 0.87, 0.88, 0.98, 0.99, 1.00])
VAL_7 = np.array([0.40, 0.42, 0.41, 0.43, 0.55, 0.54, 0.56])


class TestInterpContinuum:
    def test_linear_two_points(self):
        got = interp_continuum([0.87, 0.96], [0.40, 0.50], 0.90, "linear")
        assert got == pytest.approx(0.40 + 0.10 * 0.03 / 0.09)

    def test_single_channel_returns_value(self):
        assert interp_continuum([0.90], [0.42], 0.91) == pytest.approx(0.42)

    def test_ols_matches_polyfit_and_is_default(self):
        expected = np.polyval(np.polyfit(WL_7, VAL_7, 1), 0.91)
        assert interp_continuum(WL_7, VAL_7, 0.91) == pytest.approx(expected)

    def test_ols_extrapolates_linear_clamps(self):
        wl, val = [0.85, 0.86, 0.87], [0.40, 0.42, 0.44]
        assert interp_continuum(wl, val, 0.90, "ols") == pytest.approx(0.50)
        assert interp_continuum(wl, val, 0.90, "linear") == pytest.approx(0.44)

    def test_spline_falls_back_to_linear(self):
        args = ([0.87, 0.96], [0.40, 0.50], 0.90)
        assert interp_continuum(*args, "spline") == pytest.approx(interp_continuum(*args, "linear"))

    def test_unsorted_wavelengths(self):
        order = np.random.default_rng(0).permutation(7)
        for method in ("ols", "linear", "spline"):
            assert interp_continuum(WL_7[order], VAL_7[order], 0.91, method) == pytest.approx(
                interp_continuum(WL_7, VAL_7, 0.91, method)
            )

    def test_vectorised_matches_polyfit(self):
        val = np.random.default_rng(3).random((3, 2, 7)) * 0.3 + 0.3
        got = interp_continuum(WL_7, val, 0.91)
        for i, j in np.ndindex(3, 2):
            assert got[i, j] == pytest.approx(np.polyval(np.polyfit(WL_7, val[i, j], 1), 0.91))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown LIR"):
            interp_continuum(WL_7, VAL_7, 0.91, "cubic")


class TestComputeApdaRatio:
    def test_basic_ratio(self):
        lir = 0.40 + 0.10 * 0.04 / 0.09
        got = compute_apda_ratio([0.28, 0.32], [0.40, 0.50], [0.87, 0.96], 0.91, "linear")
        assert got == pytest.approx(0.30 / lir)

    def test_dark_target_nan(self):
        assert np.isnan(compute_apda_ratio([0.1, 0.1], [0.0, 0.0], [0.87, 0.96], 0.91))


# ---------------------------------------------------------------------------
# conversion.py
# ---------------------------------------------------------------------------

class TestConversion:
    def test_round_trip(self):
        pw = np.array([0.5, 1.0, 2.5, 4.0])
        ratio = np.exp(-(C + A * pw ** B))
        np.testing.assert_allclose(convert_ratio_to_pw(ratio, A, B, C), pw)

    def test_scalar_in_scalar_out(self):
        assert isinstance(convert_ratio_to_pw(0.8, A, B, C), float)

    def test_edge_cases(self):
        assert convert_ratio_to_pw(np.exp(-C), A, B, C) == pytest.approx(0.0)
        assert np.isnan(convert_ratio_to_pw(1.0, A, B, C))   # R > exp(-c): unphysical
        assert np.isinf(convert_ratio_to_pw(0.0, A, B, C))
        assert np.isnan(convert_ratio_to_pw(-0.1, A, B, C))

    def test_fit_recovers_params(self):
        pw = np.linspace(0.2, 6.0, 15)
        params = fit_exponential(pw, np.exp(-(C + A * pw ** B)))
        assert params == pytest.approx({"a": A, "b": B, "c": C}, rel=1e-4)

    def test_fit_too_few_points_raises(self):
        with pytest.raises(RuntimeError, match="Fewer than 3"):
            fit_exponential([1.0, 2.0, 3.0], [0.8, np.nan, -1.0])


# ---------------------------------------------------------------------------
# path_radiance.py
# ---------------------------------------------------------------------------

class TestPathRadiance:
    def _rho_atm(self, data):
        return xr.DataArray(data, dims=("WV", "wavelength"),
                            coords={"WV": [1.0, 2.0, 3.0], "wavelength": [0.9, 1.0]})

    def test_flat_rho_atm_zero_weights(self):
        g = compute_g_weights(self._rho_atm(np.full((3, 2), 0.05)))
        np.testing.assert_array_equal(g.values, 0.0)

    def test_weights_normalised(self):
        g = compute_g_weights(self._rho_atm([[0.05, 0.05], [0.06, 0.07], [0.07, 0.09]]))
        np.testing.assert_allclose(g.values, [0.5, 1.0])

    def test_scaling(self):
        da = self._rho_atm(np.full((3, 2), 0.1))
        g = xr.DataArray([0.5, 1.0], dims="wavelength", coords={"wavelength": [0.9, 1.0]})
        np.testing.assert_allclose(scale_path_radiance(da, 0.0, g).values, da.values)
        np.testing.assert_allclose(scale_path_radiance(da, -0.1, g).values[0], [0.095, 0.09])


# ---------------------------------------------------------------------------
# bands.py
# ---------------------------------------------------------------------------

class TestChannels:
    def test_curated_lookup(self):
        for band, cs in CURATED_BANDS.items():
            assert select_channels(band) is cs
            assert cs.all == cs.meas + cs.ref

    def test_unknown_band_raises(self):
        with pytest.raises(ValueError, match="No channel set"):
            select_channels("1380")

    def test_channelset_normalises_to_tuples(self):
        cs = ChannelSet("x", meas=[0.9], ref=[0.85, 1.0])
        assert cs.meas == (0.9,) and cs.ref == (0.85, 1.0)


# ---------------------------------------------------------------------------
# lut.py
# ---------------------------------------------------------------------------

class TestAPDALUT:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            APDALUT(tmp_path / "nope.nc")

    def test_validate_missing_var(self, tmp_path):
        path = tmp_path / "bad.nc"
        make_toy_lut().drop_vars("rho_toa").to_netcdf(path)
        with pytest.raises(ValueError, match="missing required"):
            APDALUT(path).validate()

    def test_select_geometry_scalar(self, lut_path):
        ds = APDALUT(lut_path()).select_geometry(44.0, 0.0, 90.0, 0.15)
        assert set(ds["rho_toa"].dims) == {"WV", "rho", "wavelength"}

    def test_select_geometry_per_pixel_clamps(self, lut_path):
        sza = xr.DataArray([[10.0, 90.0]], dims=("y", "x"))  # outside [22, 76]
        for method in ("nearest", "linear"):
            ds = APDALUT(lut_path()).select_geometry(sza, 0.0, 90.0, 0.15, method=method)
            assert {"y", "x"} <= set(ds["rho_atm"].dims)
            assert np.isfinite(ds["rho_atm"]).all()

    def test_collapse(self, lut_path):
        ds = APDALUT(lut_path()).collapse(44.0, 0.0, 90.0, 0.15, CHANNELS, rho=RHO_SURF)
        assert ds["rho_toa"].dims == ("WV", "wavelength")
        assert ds.sizes["wavelength"] == 4

    def test_collapse_unit_mismatch_raises(self, lut_path):
        nm = ChannelSet("nm", meas=(910.0, 920.0), ref=(870.0, 960.0))
        with pytest.raises(ValueError, match="not covered"):
            APDALUT(lut_path()).collapse(44.0, 0.0, 90.0, 0.15, nm)


# ---------------------------------------------------------------------------
# apda.py
# ---------------------------------------------------------------------------

class TestAPDA:
    def test_fit_recovers_params(self, lut_path):
        params = fitted_retriever(lut_path).fit_params
        assert params == pytest.approx({"a": A, "b": B, "c": C}, rel=1e-4)

    def test_fit_without_rho_selection_uses_first_nonzero(self, lut_path):
        lut = APDALUT(lut_path())
        collapsed = lut.collapse(44.0, 0.0, 90.0, 0.15, CHANNELS)  # keeps rho dim
        params = APDA(collapsed, CHANNELS).fit()
        assert params == pytest.approx({"a": A, "b": B, "c": C}, rel=1e-4)

    def test_retrieve_without_fit_raises(self, lut_path):
        collapsed = APDALUT(lut_path()).collapse(44.0, 0.0, 90.0, 0.15, CHANNELS, rho=RHO_SURF)
        with pytest.raises(RuntimeError, match="Call fit"):
            APDA(collapsed, CHANNELS).retrieve(toy_toa([[2.0]]))

    def test_rejects_per_pixel_lut(self, lut_path):
        sza = xr.DataArray([[22.0, 44.0]], dims=("y", "x"))
        collapsed = APDALUT(lut_path()).collapse(sza, 0.0, 90.0, 0.15, CHANNELS, rho=RHO_SURF)
        with pytest.raises(ValueError, match="single geometry"):
            APDA(collapsed, CHANNELS)

    def test_rejects_unknown_interp_method(self, lut_path):
        collapsed = APDALUT(lut_path()).collapse(44.0, 0.0, 90.0, 0.15, CHANNELS, rho=RHO_SURF)
        with pytest.raises(ValueError, match="interp_method"):
            APDA(collapsed, CHANNELS, interp_method="cubic")

    @pytest.mark.parametrize("method", ["ols", "linear", "spline"])
    def test_retrieve_on_and_off_grid(self, lut_path, method):
        pw_true = [[0.5, 0.75, 1.5, 2.0], [2.5, 3.5, 4.5, 5.0]]
        pw = fitted_retriever(lut_path, method).retrieve(toy_toa(pw_true))
        assert pw.dims == ("y", "x")
        assert pw.attrs["units"] == "g/cm^2"
        np.testing.assert_allclose(pw.values, pw_true, rtol=1e-2)

    def test_retrieve_pw_dependent_rho_atm(self, lut_path):
        """PW-dependent path reflectance forces interpolation between WV nodes."""
        profile = 0.05 + 0.02 * np.array(PW_GRID)
        retriever = fitted_retriever(lut_path, rho_atm=profile)
        pw_true = np.array([[1.0, 1.5, 2.5, 3.5, 4.5]])
        toa = toy_toa(pw_true, rho_atm=np.interp(pw_true, PW_GRID, profile))
        np.testing.assert_allclose(retriever.retrieve(toa, n_iter=5).values, pw_true, rtol=5e-2)

    def test_dim_order_independent(self, lut_path):
        retriever = fitted_retriever(lut_path)
        toa = toy_toa([[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]])
        a = retriever.retrieve(toa)
        b = retriever.retrieve(toa.transpose("wavelength", "x", "y"))
        np.testing.assert_allclose(b.transpose("y", "x").values, a.values)

    def test_valid_mask(self, lut_path):
        mask = xr.DataArray([[True, False]], dims=("y", "x"))
        pw = fitted_retriever(lut_path).retrieve(toy_toa([[2.0, 3.0]]), valid_mask=mask)
        assert np.isfinite(pw.values[0, 0]) and np.isnan(pw.values[0, 1])

    def test_uncovered_channel_raises(self, lut_path):
        toa = toy_toa([[2.0]]).sel(wavelength=[0.91, 0.92, 0.96])
        with pytest.raises(ValueError, match="not covered"):
            fitted_retriever(lut_path).retrieve(toa)

    def test_coarse_grid_raises(self, lut_path):
        toa = xr.DataArray(np.full((1, 1, 3), 0.3), dims=("y", "x", "wavelength"),
                           coords={"wavelength": [0.87, 0.915, 0.96]})
        with pytest.raises(ValueError, match="same input band"):
            fitted_retriever(lut_path).retrieve(toa)

    def test_optimize_path_radiance_recovers_alpha(self, lut_path):
        """Scene path reflectance scaled by Eq. 17 → σ(PW) is minimal at that α."""
        profile = 0.05 + 0.02 * np.array(PW_GRID)
        retriever = fitted_retriever(lut_path, rho_atm=profile)
        g = compute_g_weights(retriever._lut["rho_atm"])
        alpha_true = -0.1
        atm = scale_path_radiance(retriever._lut["rho_atm"], alpha_true, g).interp(WV=2.0)
        # Same PW, surfaces of different brightness.
        rho_surf = np.array([[0.1, 0.2, 0.4, 0.8]])
        t = np.exp(-(C + A * 2.0 ** B))
        ref = rho_surf + atm.sel(wavelength=0.87).item()
        meas = rho_surf * t + atm.sel(wavelength=0.91).item()
        toa = xr.DataArray(np.stack([ref, meas, meas, ref], axis=-1),
                           dims=("y", "x", "wavelength"), coords={"wavelength": list(WL)})
        alpha = retriever.optimize_path_radiance(toa, alpha_range=(-0.2, 0.0), n_iter=3)
        assert alpha == pytest.approx(alpha_true, abs=0.011)


# ---------------------------------------------------------------------------
# scene.py
# ---------------------------------------------------------------------------

class TestRetrieveScene:
    def test_per_pixel_geometry(self, lut_path):
        pw_true = [[1.0, 2.0, 3.0], [1.5, 2.5, 4.0]]
        # Off-node angles snap to the nearest LUT node (22/44/76 deg).
        sza = xr.DataArray([[20.0, 45.0, 75.0], [21.0, 43.0, 77.0]], dims=("y", "x"))
        pw = retrieve_scene(APDALUT(lut_path()), CHANNELS, toy_toa(pw_true),
                            sza=sza, vza=0.0, raa=90.0, aot=0.15, rho=RHO_SURF)
        assert pw.dims == ("y", "x")
        assert pw.attrs["n_geometry_nodes"] == 3
        np.testing.assert_allclose(pw.values, pw_true, rtol=1e-2)

    def test_geometry_dim_not_in_image_raises(self, lut_path):
        sza = xr.DataArray([22.0, 44.0], dims="t")
        with pytest.raises(ValueError, match="not present in the image"):
            retrieve_scene(APDALUT(lut_path()), CHANNELS, toy_toa([[2.0]]),
                           sza=sza, vza=0.0, raa=90.0, aot=0.15)
