"""Scene-level retrieval with per-pixel geometry.

:class:`~apda.apda.APDA` works on a LUT collapsed to a single geometry.
:func:`retrieve_scene` handles images where the geometry (``sza``, ``vza``,
``raa``, ``aot``) varies across pixels: each pixel is snapped to its nearest
LUT geometry node, and one retriever is collapsed, fitted and applied per
distinct node. The number of fits is bounded by the number of distinct
nodes in the scene, not the number of pixels.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core import log

from apda.apda import APDA
from apda.bands import ChannelSet
from apda.lut import APDALUT, GEOMETRY_DIMS


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
    """Retrieve water vapour over an image with per-pixel geometry.

    Parameters
    ----------
    lut : APDALUT
        Full (uncollapsed) LUT.
    channels : ChannelSet
        Measurement/reference channels from :func:`apda.select_channels`.
    toa_reflectance : xr.DataArray
        TOA reflectance with a ``wavelength`` dim plus any spatial dims.
    sza, vza, raa, aot : float or xr.DataArray
        Geometry, scalar or broadcastable to the image's spatial dims.
        Snapped to the nearest LUT node per pixel.
    rho : float, optional
        Surface-reflectance LUT node used to fit the model
        (see :meth:`APDALUT.collapse`).
    interp_method : {"ols", "linear", "spline"}, default "ols"
        Reference continuum method (see :class:`APDA`).
    pw_initial : float or xr.DataArray, default 2.0
        First-guess PW (g/cm²), scalar or per pixel.
    n_iter : int, default 2
        Retrieval iterations.
    valid_mask : xr.DataArray, optional
        Boolean mask over the spatial dims (True = valid). Invalid pixels → NaN.

    Returns
    -------
    xr.DataArray
        Retrieved PW (g/cm²) over the image's spatial dims. The
        ``n_geometry_nodes`` attr records how many fits were needed.
    """
    template = _spatial_template(toa_reflectance)
    spatial_dims = list(template.dims)
    shape = template.shape

    # Snap every pixel's geometry to the nearest LUT node.
    snapped = []
    for name, val in zip(GEOMETRY_DIMS, (sza, vza, raa, aot)):
        arr = _to_spatial_array(val, template, name).ravel()
        nodes = np.asarray(lut.ds[name].values, dtype=float)
        snapped.append(nodes[np.abs(arr[:, None] - nodes).argmin(axis=1)])
    keys = np.stack(snapped, axis=-1)
    nodes_used, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()

    spectra = toa_reflectance.transpose(*spatial_dims, "wavelength").values
    spectra = spectra.reshape(-1, toa_reflectance.sizes["wavelength"])
    wl_coord = toa_reflectance["wavelength"].values

    pw0 = _to_spatial_array(pw_initial, template, "pw_initial").ravel()
    out = np.full(spectra.shape[0], np.nan)

    log.debug(f"retrieve_scene: {len(nodes_used)} geometry node(s)")
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
