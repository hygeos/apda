"""Minimal APDA water-vapour retrieval demo, on the HYPS simulated dataset.

Two inputs:
- the LUT: radiative-transfer table used by the algorithm (all WV values),
- the HYPS file: MODTRAN-simulated TOA spectra, each simulated with a known
  water vapour. That known value is the truth we score against; a real
  image has no truth.

Run:  pixi run python examples/hyps_demo.py
Data: $APDA_LUT_PATH / $APDA_TOA_PATH (defaults: the HYPS files on /mnt/ceph).
"""

import os

import xarray as xr

from core import log

from apda import APDALUT, retrieve

LUT_PATH = os.environ.get("APDA_LUT_PATH", "/mnt/ceph/proj/hyperspectral/artdeco_corathyp_lut_ft4/lut_get_h2o.nc")
TOA_PATH = os.environ.get("APDA_TOA_PATH", "/mnt/ceph/proj/hyperspectral/hygeos_data_corathyp_sub_FT4.nc")

log.set_lvl(log.lvl.WARNING)  # apda logs every fit / iteration at debug level

# --- Inputs ------------------------------------------------------------------
lut = APDALUT(LUT_PATH)
lut.validate()

hyps = xr.open_dataset(TOA_PATH)
scene = xr.Dataset({
    # TOA reflectance, 3-D array of 216 × 10 × 184 = 2160 spectra:
    #   simulation_config (216): MODTRAN runs, 18 geometries × 3 WV × 4 repeats
    #   spectrum_type     (10):  the same run over 10 surfaces (snow, soils, ...)
    #   wavelength        (184): spectral bands, µm
    "rho_toa": hyps["toa_reflectance"].rename(toa_lambda0="wavelength"),
    # Geometry, 1-D arrays of 216 values (one per run). The 10 surfaces of a
    # run share its angles, so there is no spectrum_type dim: apda broadcasts.
    "sza": hyps["modtran_sun_zenith"],
    "vza": hyps["modtran_view_zenith"],
    "saa": hyps["modtran_sun_azimuth"],
    "vaa": hyps["modtran_view_azimuth"],
    "aot": hyps["modtran_aot"],
})
# truth: the water vapour each config was simulated with, kg/m² → g/cm²
wv_true = hyps["modtran_water_vapor"] * 0.1  # (simulation_config,)

# --- Retrieval ---------------------------------------------------------------
for band in ("910", "1100"):
    # rho=0.5: LUT surface-reflectance node the model is fitted on
    pw = retrieve(lut, scene, band, rho=0.5)  # (simulation_config, spectrum_type), g/cm²

    # --- Scoring: truth is per config, broadcast over the 10 surfaces --------
    rel_err = abs(pw - wv_true) / wv_true
    print(f"{band} nm: median relative error {float(rel_err.median()):.1%}")
