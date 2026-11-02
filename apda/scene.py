"""Scene-level retrieval with per-pixel geometry.

:class:`~apda.apda.APDA` works on a LUT collapsed to a single geometry.
:func:`retrieve` handles scenes (images or sets of simulations) where the
geometry (``sza``, ``vza``, ``raa``, ``aot``) varies across pixels. Pixel
geometry is rounded to :data:`GEOMETRY_STEP`, and one retriever is collapsed
(LUT linearly interpolated at that geometry), fitted and applied per
distinct value. The number of fits is bounded by the
number of distinct geometries in the scene, not the number of pixels.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from core import log

from apda.apda import APDA
from apda.bands import ChannelSet, select_channels
from apda.lut import APDALUT, GEOMETRY_DIMS, clip_to_range

#: Resolution at which pixel geometry is binned before the LUT is
#: interpolated: one fit per distinct bin.
#: Halving a step at most doubles the number of fits along that dim.
GEOMETRY_STEP: dict[str, float] = {"sza": 0.1, "vza": 0.1, "raa": 1.0, "aot": 0.01}


def _spatial_template(toa_reflectance: xr.DataArray) -> xr.DataArray:
    """A (spatial dims only) template of the image, used for broadcasting."""
    return toa_reflectance.isel(wavelength=0, drop=True)


def _to_spatial_array(
    val: float | xr.DataArray, template: xr.DataArray, name: str
) -> np.ndarray:
    """Broadcast a scalar or DataArray onto the image's spatial grid."""
    da = val if isinstance(val, xr.DataArray) else xr.DataArray(float(val))
    extra = set(da.dims) - set(template.dims)
    if extra:
        raise ValueError(
            f"'{name}' has dims {sorted(extra)} not present in the image "
            f"(spatial dims: {list(template.dims)})."
        )
    return da.broadcast_like(template).transpose(*template.dims).values


#: Variable names :func:`retrieve` reads from the scene Dataset.
REFLECTANCE_VAR: str = "rho_toa"
WV_INIT_VAR: str = "wv_init"
VALID_VAR: str = "valid"
_ANGLES: tuple[str, ...] = ("sza", "vza", "raa", "saa", "vaa")
_RADIANS: tuple[str, ...] = ("rad", "radian", "radians")


def _fold_azimuth(dphi: xr.DataArray) -> xr.DataArray:
    """Fold an azimuth difference into [0, 180] deg (LUT ``raa`` convention)."""
    dphi = np.abs(dphi) % 360
    return np.minimum(dphi, 360 - dphi)


def scene_geometry(scene: xr.Dataset) -> dict[str, xr.DataArray]:
    """``sza, vza, raa, aot`` from a scene Dataset (see :func:`retrieve`).

    ``raa`` is taken from ``raa`` if present, else computed from ``saa`` and
    ``vaa``; either way it is folded into [0, 180] deg. Raises ``ValueError``
    on missing variables or angles whose ``units`` attr says radians.
    """
    for name in _ANGLES:
        if name in scene and str(scene[name].attrs.get("units", "")).lower() in _RADIANS:
            raise ValueError(f"'{name}' is in radians; apda expects degrees.")
    if "raa" in scene:
        raa = _fold_azimuth(scene["raa"])
    elif "saa" in scene and "vaa" in scene:
        raa = _fold_azimuth(scene["saa"] - scene["vaa"])
    else:
        raise ValueError("Scene needs 'raa', or 'saa' and 'vaa' (degrees).")
    missing = [v for v in ("sza", "vza", "aot") if v not in scene]
    if missing:
        raise ValueError(f"Scene missing geometry variables: {missing}")
    return {"sza": scene["sza"], "vza": scene["vza"], "raa": raa, "aot": scene["aot"]}


def retrieve(
    lut: APDALUT,
    scene: xr.Dataset,
    band: str | ChannelSet,
    rho: float | None = None,
    interp_method: str = "ols",
    n_iter: int = 2,
) -> xr.DataArray:
    """Retrieve water vapour over a scene: an image or a set of simulations.

    The scene is an ``xr.Dataset`` with:

    ============  ==============================  =========================
    variable      dims                            units
    ============  ==============================  =========================
    ``rho_toa``   ``(..., wavelength)``           reflectance; wavelength
                                                  in the LUT unit (µm)
    ``sza``,      scalar or a subset of the       degrees
    ``vza``       spatial dims
    ``raa`` or    same                            degrees, any range
    ``saa`` +                                     (folded into [0, 180])
    ``vaa``
    ``aot``       same                            —
    ``wv_init``   same, optional (default 2.0)    g/cm²
    ``valid``     same, optional                  bool, False → NaN
    ============  ==============================  =========================

    The spatial dims are whatever ``rho_toa`` has besides ``wavelength``:
    ``(y, x)`` for an image, ``(simulation_config, ...)`` for simulations.
    Other variables, coordinates and attributes are ignored.

    Parameters
    ----------
    lut : APDALUT
        Full (uncollapsed) LUT.
    scene : xr.Dataset
        Input, see above.
    band : str or ChannelSet
        ``"910"``, ``"1100"`` (see :func:`apda.select_channels`) or a custom
        channel set.
    rho : float, optional
        Surface-reflectance LUT node used to fit the model
        (see :meth:`APDALUT.collapse`).
    interp_method : {"ols", "linear", "spline"}, default "ols"
        Reference continuum method (see :class:`APDA`).
    n_iter : int, default 2
        Retrieval iterations.

    Returns
    -------
    xr.DataArray
        ``water_vapour_column`` (g/cm²) over the spatial dims, with the
        scene's spatial coordinates. The LUT is linearly interpolated at each
        pixel's geometry, rounded to :data:`GEOMETRY_STEP`; geometry outside
        the LUT range is clamped with a warning; pixels with NaN geometry
        come out NaN. ``n_geometry_nodes`` records how many fits were needed.

    Examples
    --------
    >>> scene = xr.Dataset({"rho_toa": toa, "sza": 44.0, "vza": 0.0,
    ...                     "saa": saa, "vaa": vaa, "aot": 0.15})
    >>> pw = retrieve(lut, scene, "910")
    """
    if REFLECTANCE_VAR not in scene or "wavelength" not in scene[REFLECTANCE_VAR].dims:
        raise ValueError(f"Scene needs a '{REFLECTANCE_VAR}' variable with a 'wavelength' dim.")
    channels = select_channels(band) if isinstance(band, str) else band
    return _retrieve(
        lut, channels, scene[REFLECTANCE_VAR], scene_geometry(scene), rho=rho,
        interp_method=interp_method,
        pw_initial=scene[WV_INIT_VAR] if WV_INIT_VAR in scene else 2.0,
        n_iter=n_iter,
        valid_mask=scene[VALID_VAR] if VALID_VAR in scene else None,
    )


def retrieve_scene(
    lut: APDALUT,
    channels: ChannelSet,
    toa_reflectance: xr.DataArray,
    sza: float | xr.DataArray,
    vza: float | xr.DataArray,
    raa: float | xr.DataArray,
    aot: float | xr.DataArray,
    rho: float | None = None,
    interp_method: str = "ols",
    pw_initial: float | xr.DataArray = 2.0,
    n_iter: int = 2,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Deprecated: use :func:`retrieve` with a scene Dataset.

    Same retrieval, with the geometry passed as separate arguments and
    ``raa`` expected in [0, 180] deg.
    """
    warnings.warn("retrieve_scene is deprecated; use apda.retrieve(lut, scene, band).",
                  DeprecationWarning, stacklevel=2)
    geometry = {"sza": sza, "vza": vza, "raa": raa, "aot": aot}
    return _retrieve(lut, channels, toa_reflectance, geometry, rho, interp_method,
                     pw_initial, n_iter, valid_mask)


def _retrieve(
    lut: APDALUT,
    channels: ChannelSet,
    toa_reflectance: xr.DataArray,
    geometry: dict,
    rho: float | None = None,
    interp_method: str = "ols",
    pw_initial: float | xr.DataArray = 2.0,
    n_iter: int = 2,
    valid_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Per-geometry retrieval shared by :func:`retrieve` and :func:`retrieve_scene`."""
    template = _spatial_template(toa_reflectance)
    spatial_dims = list(template.dims)
    shape = template.shape

    # Clamp (with a warning) and bin every pixel's geometry; NaN stays NaN.
    binned = []
    for name in GEOMETRY_DIMS:
        lo, hi = float(lut.ds[name].min()), float(lut.ds[name].max())
        arr = clip_to_range(_to_spatial_array(geometry[name], template, name).ravel(), lo, hi, name)
        step = GEOMETRY_STEP[name]
        # round twice so on-node values (multiples of step) stay exact
        binned.append(np.clip(np.round(np.round(arr / step) * step, 10), lo, hi))
    keys = np.stack(binned, axis=-1)
    ok = ~np.isnan(keys).any(axis=1)
    nodes_used, group_ok = np.unique(keys[ok], axis=0, return_inverse=True)
    group = np.full(keys.shape[0], -1)
    group[ok] = group_ok.ravel()

    spectra = toa_reflectance.transpose(*spatial_dims, "wavelength").values
    spectra = spectra.reshape(-1, toa_reflectance.sizes["wavelength"])
    wl_coord = toa_reflectance["wavelength"].values

    pw0 = _to_spatial_array(pw_initial, template, "pw_initial").ravel()
    out = np.full(spectra.shape[0], np.nan)

    log.debug(f"retrieve: {len(nodes_used)} distinct geometries")
    for k, node in enumerate(nodes_used):
        idx = np.flatnonzero(group == k)
        geom = dict(zip(GEOMETRY_DIMS, map(float, node)))
        collapsed = lut.collapse(**geom, channels=channels, rho=rho)
        retriever = APDA(collapsed, channels, interp_method=interp_method)
        retriever.fit()

        pix = xr.DataArray(
            spectra[idx], dims=("pixel", "wavelength"),
            coords={"wavelength": wl_coord},
        )
        pw_k = xr.DataArray(pw0[idx], dims="pixel")
        out[idx] = retriever.retrieve(pix, pw_initial=pw_k, n_iter=n_iter).values

    pw = template.copy(data=out.reshape(shape))
    if valid_mask is not None:
        pw = pw.where(valid_mask)
    pw.name = "water_vapour_column"
    pw.attrs = {
        "units": "g/cm^2",
        "description": f"Columnar H2O retrieved by APDA ({n_iter} iterations)",
        "band": channels.band,
        "n_geometry_nodes": len(nodes_used),
    }
    return pw
