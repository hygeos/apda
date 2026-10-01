"""Integration tests on the real HYPS LUT + simulated TOA dataset.

These tests validate the full APDA pipeline end-to-end:

1. Load the operational LUT (`lut_get_h2o.nc`).
2. Collapse geometry + select wavelengths, per geometry node.
3. Fit a, b, c (Eq. 14).
4. Retrieve WV on the TOA dataset.
5. Validate vs `modtran_water_vapor × 0.1` (kg/m² → g/cm²).

Dataset paths are read from the ``APDA_LUT_PATH`` and ``APDA_TOA_PATH``
environment variables; tests are **skipped** when they are unset or the
files are not accessible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from apda import APDA, APDALUT, retrieve_scene, select_channels


LUT_PATH = os.environ.get("APDA_LUT_PATH", "")
TOA_PATH = os.environ.get("APDA_TOA_PATH", "")

skip_no_lut = pytest.mark.skipif(
    not Path(LUT_PATH).is_file(),
    reason="Set APDA_LUT_PATH to the HYPS LUT to run integration tests.",
)
skip_no_toa = pytest.mark.skipif(
    not Path(TOA_PATH).is_file(),
    reason="Set APDA_TOA_PATH to the HYPS TOA dataset to run integration tests.",
)

#: Median relative error threshold on the first surface spectrum.
MAX_MEDIAN_ERR = 0.10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lut_wrapper():
    """Load and validate the LUT once per module."""
    wrapper = APDALUT(LUT_PATH)
    wrapper.validate()
    return wrapper


@pytest.fixture(scope="module")
def toa_dataset():
    """Load the TOA dataset."""
    return xr.open_dataset(TOA_PATH)


@pytest.fixture(scope="module")
def scene(toa_dataset):
    """First surface spectrum as a scene, per-config geometry, truth WV."""
    ds = toa_dataset
    toa = (ds["toa_reflectance"].isel(spectrum_type=0)
           .rename({"toa_lambda0": "wavelength"}))
    dphi = np.abs(ds["modtran_sun_azimuth"] - ds["modtran_view_azimuth"]) % 360
    geometry = {
        "sza": ds["modtran_sun_zenith"],
        "vza": ds["modtran_view_zenith"],
        "raa": np.minimum(dphi, 360 - dphi),
        "aot": ds["modtran_aot"],
    }
    wv_true = ds["modtran_water_vapor"].values * 0.1
    return toa, geometry, wv_true


# ---------------------------------------------------------------------------
# Tests — LUT loading and fit
# ---------------------------------------------------------------------------

@skip_no_lut
class TestLUTFit:
    """Test LUT loading, geometry collapse, and exponential fit."""

    def test_lut_loads_and_validates(self, lut_wrapper):
        """LUT loads and passes validation."""
        assert "rho_atm" in lut_wrapper.ds.data_vars
        assert "rho_toa" in lut_wrapper.ds.data_vars

    @pytest.mark.parametrize("band", ["910", "1100"])
    def test_fit_params_positive(self, lut_wrapper, band):
        """Fitted a, b are positive (ratio decreases with WV)."""
        channels = select_channels(band)
        collapsed = lut_wrapper.collapse(
            sza=44.0, vza=0.0, raa=90.0, aot=0.15,
            channels=channels, rho=0.5,
        )
        params = APDA(collapsed, channels).fit()
        assert params["a"] > 0, f"a={params['a']} should be positive"
        assert params["b"] > 0, f"b={params['b']} should be positive"


# ---------------------------------------------------------------------------
# Tests — Retrieval and validation
# ---------------------------------------------------------------------------

@skip_no_lut
@skip_no_toa
class TestRetrieval:
    """Test WV retrieval and validation against ground truth."""

    @pytest.mark.parametrize("band", ["910", "1100"])
    def test_retrieve_scene(self, lut_wrapper, scene, band):
        """Per-geometry retrieval over all configs is within tolerance."""
        toa, geometry, wv_true = scene
        pw = retrieve_scene(lut_wrapper, select_channels(band), toa,
                            **geometry, rho=0.5, n_iter=2)
        est = pw.values

        valid = np.isfinite(est) & (wv_true > 0)
        assert valid.all(), f"{band} nm: {(~valid).sum()} invalid retrievals"
        median_err = float(np.median(np.abs(est - wv_true) / wv_true))
        assert median_err < MAX_MEDIAN_ERR, (
            f"{band} nm: median relative error {median_err:.1%} "
            f"> {MAX_MEDIAN_ERR:.0%}"
        )
