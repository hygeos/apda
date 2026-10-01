"""LUT wrapper — loading, validation, geometry and spectral collapse.

The LUT is an operational MODTRAN-style NetCDF file in **reflectance
space** (dimensionless) with variables ``rho_atm`` (path reflectance) and
``rho_toa`` (TOA reflectance), and coordinates ``WV`` (g/cm²),
``wavelength`` (µm for HYPS), geometry (``sza``, ``vza``, ``raa`` in deg,
``aot``) and surface reflectance ``rho``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from core import log

from apda.bands import ChannelSet

#: Data variables used by the retrieval.
REQUIRED_VARS: tuple[str, ...] = ("rho_atm", "rho_toa")

#: Coordinates the LUT must provide.
REQUIRED_COORDS: tuple[str, ...] = ("WV", "wavelength", "sza", "vza", "raa", "aot", "rho")

#: Name of the water-vapour (retrieval) dimension, in g/cm².
PW_DIM: str = "WV"

#: Geometry dimensions collapsed by :meth:`APDALUT.select_geometry`.
GEOMETRY_DIMS: tuple[str, ...] = ("sza", "vza", "raa", "aot")

#: Max distance between a requested wavelength and the nearest available
#: band, in wavelength-coordinate units (µm for HYPS).
WL_TOLERANCE: float = 0.01


def select_wavelengths(obj, wavelengths) -> xr.Dataset | xr.DataArray:
    """Nearest-wavelength selection with coverage checks.

    Raises ``ValueError`` if a requested wavelength has no band within
    :data:`WL_TOLERANCE` (missing coverage or unit mismatch), or if two
    requested wavelengths map to the same band (grid too coarse).
    """
    wl = np.asarray(wavelengths, dtype=float)
    sel = obj.sortby("wavelength").sel(wavelength=wl, method="nearest")
    got = np.asarray(sel["wavelength"].values, dtype=float)

    off = np.abs(got - wl) > WL_TOLERANCE
    if off.any():
        raise ValueError(
            f"Wavelengths {wl[off].tolist()} are not covered by the input "
            f"spectral grid (tolerance {WL_TOLERANCE:g}). Check units and "
            "spectral coverage."
        )
    if len(np.unique(got)) != len(got):
        raise ValueError(
            f"Several wavelengths in {wl.tolist()} map to the same input band "
            f"({got.tolist()}); the input spectral grid is too coarse."
        )
    return sel


class APDALUT:
    """Operational reflectance LUT.

    Parameters
    ----------
    path : str or Path
        NetCDF LUT file (opened lazily by xarray).

    Examples
    --------
    >>> lut = APDALUT("lut_get_h2o.nc")
    >>> lut.validate()
    >>> collapsed = lut.collapse(44.0, 0.0, 90.0, 0.15, select_channels("910"), rho=0.5)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"LUT file not found: {self.path}")
        self.ds: xr.Dataset = xr.open_dataset(self.path)
        log.debug(f"Loaded LUT: {self.path.name} ({dict(self.ds.sizes)})")

    def validate(self) -> None:
        """Check required variables/coordinates and that ``WV`` is increasing.

        Raises
        ------
        ValueError
            On missing variables/coordinates or a non-monotonic ``WV`` axis.
        """
        missing = [v for v in REQUIRED_VARS if v not in self.ds.data_vars]
        missing += [c for c in REQUIRED_COORDS if c not in self.ds.coords]
        if missing:
            raise ValueError(f"LUT missing required variables/coordinates: {missing}")

        wv = np.asarray(self.ds[PW_DIM].values, dtype=float)
        if not np.all(np.diff(wv) > 0):
            raise ValueError(f"LUT '{PW_DIM}' must be strictly increasing, got {wv}")

    def select_geometry(
        self,
        sza: float | xr.DataArray,
        vza: float | xr.DataArray,
        raa: float | xr.DataArray,
        aot: float | xr.DataArray,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Collapse the geometry dimensions.

        Parameters
        ----------
        sza, vza, raa, aot : float or xr.DataArray
            Geometry. Scalars give a single-geometry LUT; DataArrays (e.g.
            dims ``(y, x)``) give a per-pixel LUT with those dims added.
            Values are clamped to the LUT range.
        method : {"nearest", "linear"}, default "nearest"
            ``"nearest"`` is exact when the geometry lies on the LUT grid;
            ``"linear"`` interpolates between nodes.

        Returns
        -------
        xr.Dataset
            LUT without the geometry dims.
        """
        if method not in ("nearest", "linear"):
            raise ValueError(f"Unknown geometry method '{method}'. Use 'nearest' or 'linear'.")

        geom = {}
        for name, val in zip(GEOMETRY_DIMS, (sza, vza, raa, aot)):
            lo, hi = float(self.ds[name].min()), float(self.ds[name].max())
            geom[name] = val.clip(lo, hi) if isinstance(val, xr.DataArray) else float(np.clip(val, lo, hi))

        if method == "nearest" and not any(isinstance(v, xr.DataArray) for v in geom.values()):
            return self.ds.sel(geom, method="nearest")
        return self.ds.interp(geom, method=method)

    def collapse(
        self,
        sza: float | xr.DataArray,
        vza: float | xr.DataArray,
        raa: float | xr.DataArray,
        aot: float | xr.DataArray,
        channels: ChannelSet,
        rho: float | None = None,
        geom_method: str = "nearest",
    ) -> xr.Dataset:
        """Collapse geometry, keep only ``channels``, optionally pick a ``rho`` node.

        Parameters
        ----------
        sza, vza, raa, aot : float or xr.DataArray
            Geometry (see :meth:`select_geometry`).
        channels : ChannelSet
            Channels to keep (from :func:`apda.select_channels`).
        rho : float, optional
            Surface-reflectance node used to fit the model (nearest LUT
            value). If None, :meth:`APDA.fit` uses the first non-zero node.
        geom_method : {"nearest", "linear"}, default "nearest"
            Geometry interpolation (see :meth:`select_geometry`).

        Returns
        -------
        xr.Dataset
            Compact LUT with dims ``(WV, wavelength)``, the input of
            :class:`apda.APDA`.
        """
        ds = self.select_geometry(sza, vza, raa, aot, method=geom_method)
        ds = select_wavelengths(ds, channels.all)
        if rho is not None:
            ds = ds.sel(rho=rho, method="nearest")
        return ds
