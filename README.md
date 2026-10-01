# apda

Columnar water-vapour retrieval from hyperspectral imagery with the
**Atmospheric Precorrected Differential Absorption** (APDA) method
(Schläpfer et al., 1998), using a pre-computed radiative-transfer look-up
table (LUT).

The library works in reflectance space: the E₀/π factor cancels in the APDA
ratio. It reads operational MODTRAN-style LUTs (`rho_atm`, `rho_toa` over
`WV`, `wavelength`, `sza`, `vza`, `raa`, `aot`, `rho`) and fits the
ratio-to-water-vapour model directly from the LUT reflectances.

## Installation

Python ≥ 3.10.

```bash
pip install /path/to/apda          # or: pip install git+<repo-url>
```

For development with [pixi](https://pixi.sh):

```bash
pixi install
pixi run tests
```

## Usage

```python
from apda import APDALUT, retrieve_scene, select_channels

lut = APDALUT("lut_get_h2o.nc")
lut.validate()

# toa: xr.DataArray of TOA reflectance with a `wavelength` dim (µm, same unit
#      as the LUT) plus any spatial dims, e.g. (y, x, wavelength).
# sza, vza, raa, aot: scalars or DataArrays over the spatial dims
#      (raa in [0, 180] deg, LUT convention).
pw = retrieve_scene(lut, select_channels("910"), toa,
                    sza=sza, vza=vza, raa=raa, aot=aot, rho=0.5)
# → water_vapour_column (g/cm²) over the spatial dims
```

`retrieve_scene` snaps each pixel to its nearest LUT geometry node and fits
one model per distinct node. For a single geometry, the building blocks are:

```python
from apda import APDA

channels = select_channels("1100")
collapsed = lut.collapse(sza=44.0, vza=0.0, raa=90.0, aot=0.15,
                         channels=channels, rho=0.5)
retriever = APDA(collapsed, channels)
retriever.fit()                          # {"a": …, "b": …, "c": …}
pw = retriever.retrieve(toa, n_iter=2)

# Optional empirical path-reflectance tuning (Eqs. 16–17) on a flat,
# spectrally diverse subset:
alpha = retriever.optimize_path_radiance(toa, subset_mask=mask)
pw = retriever.retrieve(toa, alpha=alpha)
```

Channels are matched to the image and LUT by nearest wavelength. A
`ValueError` is raised if a channel has no band within 0.01 (LUT units), which
usually means a unit mismatch or missing spectral coverage, or if two channels
fall on the same band.

## API

| Name | Role |
|---|---|
| `APDALUT` | Load / validate a LUT; `select_geometry`, `collapse` |
| `select_channels`, `ChannelSet` | Measurement/reference channel sets (`"910"`, `"1100"`) |
| `APDA` | Single-geometry retriever: `fit`, `compute_ratio`, `retrieve`, `optimize_path_radiance` |
| `retrieve_scene` | Image retrieval with per-pixel geometry |
| `compute_apda_ratio` | APDA ratio (Eq. 6) |
| `fit_exponential`, `convert_ratio_to_pw` | Ratio ↔ PW model `R = exp(-(c + a·PW^b))` (Eqs. 14–15) |

Full parameter descriptions are in the docstrings.

## Validation

On the HYPS simulated dataset (216 configs = 18 geometries × 3 WV × 4
repeats, × 10 surface spectra; `rho=0.5`, 2 iterations), the median relative
error against the MODTRAN water-vapour truth is:

| Band | Median | Mean | Median per WV (0.87 / 2.98 / 4.20 g/cm²) |
|---|---|---|---|
| 910 nm | 4.4% | 5.5% | 4.8% / 4.2% / 4.2% |
| 1100 nm | 10.2% | 13.5% | 14.8% / 8.0% / 8.3% |

At 1100 nm, accuracy depends strongly on the surface (2.8% on Gobabeb, 31% on
snow); at 910 nm, every surface stays within 2–9%. The data are simulated and
noise-free.

`tests/test_integration_hyps.py` runs on this dataset when `APDA_LUT_PATH` and
`APDA_TOA_PATH` are set.

## Reference

Schläpfer, D., Borel, C. C., Keller, J., and Itten, K. I. (1998). Atmospheric
Precorrected Differential Absorption Technique to Retrieve Columnar Water
Vapor. *Remote Sensing of Environment*, 65, 353–366.

---

[HYGEOS](https://hygeos.com/en/)
