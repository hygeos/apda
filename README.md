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

The input is a **scene**: an `xr.Dataset` describing either an image or a set
of simulations.

| Variable | Dims | Units |
|---|---|---|
| `rho_toa` | `(..., wavelength)` | TOA reflectance; wavelength in the LUT unit (µm) |
| `sza`, `vza` | scalar or a subset of the spatial dims | degrees |
| `raa`, or `saa` + `vaa` | same | degrees, any range (folded into [0, 180]) |
| `aot` | same | — |
| `wv_init` *(optional)* | same | first-guess PW, g/cm² (default 2.0) |
| `valid` *(optional)* | same | bool, `False` → NaN |

The spatial dims are whatever `rho_toa` has besides `wavelength`: `(y, x)` for
an image, `(simulation_config, ...)` for simulations. Other variables,
coordinates and attributes are ignored; angles whose `units` attribute says
radians are rejected.

```python
import xarray as xr
from apda import APDALUT, retrieve

lut = APDALUT("lut_get_h2o.nc")
lut.validate()

scene = xr.Dataset({"rho_toa": toa, "sza": sza, "vza": vza,
                    "saa": saa, "vaa": vaa, "aot": 0.15})
pw = retrieve(lut, scene, "910", rho=0.5)
# → water_vapour_column (g/cm²) over the spatial dims, with the scene's coords
```

`retrieve` linearly interpolates the LUT at each pixel's geometry, rounded to
`apda.scene.GEOMETRY_STEP` (0.1° for `sza`/`vza`, 1° for `raa`, 0.01 for
`aot`), and fits one model per distinct value. Geometry outside the LUT range
is clamped to it with a warning; NaN geometry gives NaN. See
`examples/hyps_demo.py` for a complete run on the HYPS dataset.

### Choosing `rho`

The ratio → water-vapour model is fitted on LUT spectra simulated over a flat
surface of reflectance `rho` (a LUT node: 0.3, 0.5 or 1.0 for HYPS). It is an
algorithm setting, the same for every pixel, so pass it explicitly. All
validation figures use `rho=0.5`. Without it, the first non-zero LUT node is
used.

The choice matters little. On HYPS (median of per-surface median errors),
even picking the best `rho` for each surface in hindsight gains about one
point at 910 nm and almost nothing at 1100 nm, where the error comes from the
surface's spectral shape, not its brightness:

| Band | `rho=0.5` | best `rho` per surface |
|---|---|---|
| 910 nm | 4.3% | 3.2% |
| 1100 nm | 8.9% | 8.4% |

For a single geometry, the building blocks are:

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
| `retrieve` | Retrieval over a scene Dataset (image or simulations), per-pixel geometry |
| `scene_geometry` | The `sza, vza, raa, aot` `retrieve` reads from a scene |
| `retrieve_scene` | *Deprecated*: `retrieve` with geometry as separate arguments |
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
