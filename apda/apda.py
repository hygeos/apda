"""APDA retriever for a single geometry.

Pipeline (Schläpfer et al. 1998, RSE 65:353-366):

1. :meth:`APDA.fit`           — fit a, b, c from the LUT (Eq. 14).
2. :meth:`APDA.compute_ratio` — APDA ratio image (Eq. 6).
3. :meth:`APDA.retrieve`      — iterative PW retrieval (Eq. 15).
4. :meth:`APDA.optimize_path_radiance` — optional α tuning (Eqs. 16–17).

Synthetic ratios are computed directly from the LUT's reflectances,

    R_APDA(WV) = mean(rho_toa_m - rho_atm_m) / LIR(rho_toa_r - rho_atm_r)

instead of reconstructing ``L = L_gnd·T_wv + L_atm``, which avoids
deriving a water-vapour transmittance. Everything stays in reflectance
space — the E₀/π factor cancels in the ratio.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core import log

from apda.bands import ChannelSet
from apda.conversion import convert_ratio_to_pw, fit_exponential
from apda.lut import PW_DIM, select_wavelengths
from apda.path_radiance import compute_g_weights, scale_path_radiance
from apda.ratio import LIR_METHODS, compute_apda_ratio


class APDA:
    """Atmospheric Precorrected Differential Absorption retriever.

    Parameters
    ----------
    lut : xr.Dataset
        LUT collapsed to a single geometry (:meth:`APDALUT.collapse` with
        scalar angles): ``rho_atm`` and ``rho_toa`` over ``(WV, wavelength)``
        and optionally ``rho``. For per-pixel geometry use
        :func:`apda.retrieve_scene`.
    channels : ChannelSet
        Measurement/reference channels — the same set used to collapse the LUT.
    interp_method : {"ols", "linear", "spline"}, default "ols"
        Reference continuum method (see :func:`apda.ratio.interp_continuum`).

    Attributes
    ----------
    fit_params : dict[str, float]
        ``{"a": …, "b": …, "c": …}``, populated by :meth:`fit`.
    """

    def __init__(
        self,
        lut: xr.Dataset,
        channels: ChannelSet,
        interp_method: str = "ols",
    ) -> None:
        missing = {"rho_atm", "rho_toa"} - set(lut.data_vars)
        if missing:
            raise ValueError(f"LUT missing required variables: {sorted(missing)}")
        extra = set(lut["rho_atm"].dims) - {PW_DIM, "wavelength"}
        if PW_DIM not in lut["rho_atm"].dims or extra:
            raise ValueError(
                f"APDA expects a LUT collapsed to a single geometry, with "
                f"rho_atm over ({PW_DIM!r}, 'wavelength'); got {lut['rho_atm'].dims}. "
                "For varying geometry use apda.retrieve_scene."
            )
        if interp_method not in LIR_METHODS:
            raise ValueError(f"Unknown interp_method '{interp_method}'. Use one of {LIR_METHODS}.")

        self._lut = lut
        self.channels = channels
        self._meas_wl = np.sort(channels.meas)
        self._ref_wl = np.sort(channels.ref)
        self._lambda_m = float(self._meas_wl.mean())
        self._interp_method = interp_method
        self.fit_params: dict[str, float] = {}

    # ------------------------------------------------------------------

    def _path_reflectance(
        self,
        pw: float | xr.DataArray,
        wavelengths: np.ndarray,
        rho_atm: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """rho_atm at ``wavelengths``, linearly interpolated to ``pw`` (clamped)."""
        var = select_wavelengths(self._lut["rho_atm"] if rho_atm is None else rho_atm, wavelengths)
        lo, hi = float(var[PW_DIM].min()), float(var[PW_DIM].max())
        pw = pw.clip(lo, hi) if isinstance(pw, xr.DataArray) else float(np.clip(pw, lo, hi))
        return var.interp({PW_DIM: pw})

    def fit(self) -> dict[str, float]:
        """Fit ``R = exp(-(c + a·PW^b))`` to synthetic LUT ratios (Eq. 14).

        If the LUT still has a ``rho`` dim, its first non-zero node is used.

        Returns
        -------
        dict[str, float]
            ``{"a": …, "b": …, "c": …}`` — also stored in :attr:`fit_params`.
        """
        wl = np.r_[self._meas_wl, self._ref_wl]
        rho_toa = select_wavelengths(self._lut["rho_toa"], wl)
        if "rho" in rho_toa.dims:
            rho = rho_toa["rho"].values
            rho_toa = rho_toa.sel(rho=rho[rho != 0][0])
        rho_atm = select_wavelengths(self._lut["rho_atm"], wl)

        corr = (rho_toa - rho_atm).transpose(PW_DIM, "wavelength").values
        n = len(self._meas_wl)
        ratios = compute_apda_ratio(
            corr[:, :n], corr[:, n:], self._ref_wl, self._lambda_m, self._interp_method
        )
        self.fit_params = fit_exponential(self._lut[PW_DIM].values, ratios)
        return self.fit_params

    def compute_ratio(
        self,
        toa_reflectance: xr.DataArray,
        pw: float | xr.DataArray,
        rho_atm: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """APDA ratio image (Eq. 6).

        Parameters
        ----------
        toa_reflectance : xr.DataArray
            TOA reflectance with a ``wavelength`` dim (same unit as the LUT)
            plus any spatial dims, in any order.
        pw : float or xr.DataArray
            Current PW estimate (g/cm²) for the path-reflectance look-up.
        rho_atm : xr.DataArray, optional
            Replacement path reflectance (e.g. Eq. 17-scaled). Default: LUT's.

        Returns
        -------
        xr.DataArray
            Dimensionless ratio over the spatial dims; NaN on dark targets.
        """
        corr = []
        for wl in (self._meas_wl, self._ref_wl):
            toa = select_wavelengths(toa_reflectance, wl)
            atm = self._path_reflectance(pw, wl, rho_atm)
            # Match by position: nearest selection may leave slightly
            # different wavelength coords on the image and the LUT.
            atm = atm.assign_coords(wavelength=toa["wavelength"].values)
            corr.append((toa - atm).transpose(..., "wavelength"))
        corr_m, corr_r = corr
        corr_m = corr_m.transpose(*corr_r.dims)

        ratio = compute_apda_ratio(
            corr_m.values, corr_r.values, self._ref_wl, self._lambda_m, self._interp_method
        )
        out = corr_r.isel(wavelength=0, drop=True).copy(data=ratio)
        out.name = "apda_ratio"
        out.attrs = {"units": "dimensionless", "description": "APDA ratio (Eq. 6)"}
        return out

    def retrieve(
        self,
        toa_reflectance: xr.DataArray,
        pw_initial: float | xr.DataArray = 2.0,
        n_iter: int = 2,
        alpha: float | None = None,
        valid_mask: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Retrieve the water-vapour column for every pixel (Eq. 15).

        Each iteration looks up the path reflectance at the current PW,
        recomputes the ratio and inverts it; two iterations are typically
        within 1% of convergence.

        Parameters
        ----------
        toa_reflectance : xr.DataArray
            TOA reflectance (see :meth:`compute_ratio`).
        pw_initial : float or xr.DataArray, default 2.0
            First-guess PW (g/cm²).
        n_iter : int, default 2
            Number of iterations.
        alpha : float, optional
            Apply the Eq. 17 path-reflectance scaling with this α.
        valid_mask : xr.DataArray, optional
            Boolean mask (True = valid); invalid pixels → NaN.

        Returns
        -------
        xr.DataArray
            ``water_vapour_column`` (g/cm²) over the spatial dims.
        """
        if not self.fit_params:
            raise RuntimeError("Call fit() before retrieve().")

        rho_atm = None
        if alpha is not None:
            g = compute_g_weights(self._lut["rho_atm"], pw_dim=PW_DIM)
            rho_atm = scale_path_radiance(self._lut["rho_atm"], alpha, g)

        pw = pw_initial
        for i in range(n_iter):
            log.debug(f"APDA iteration {i + 1}/{n_iter}")
            ratio = self.compute_ratio(toa_reflectance, pw, rho_atm)
            pw = ratio.copy(data=convert_ratio_to_pw(ratio.values, **self.fit_params))
            if valid_mask is not None:
                pw = pw.where(valid_mask)

        pw.name = "water_vapour_column"
        pw.attrs = {
            "units": "g/cm^2",
            "description": f"Columnar H2O retrieved by APDA ({n_iter} iterations)",
            "band": self.channels.band,
            **{f"fit_{k}": v for k, v in self.fit_params.items()},
        }
        if alpha is not None:
            pw.attrs["alpha"] = alpha
        return pw

    def optimize_path_radiance(
        self,
        toa_reflectance: xr.DataArray,
        subset_mask: xr.DataArray | None = None,
        pw_initial: float | xr.DataArray = 2.0,
        n_iter: int = 2,
        alpha_range: tuple[float, float] = (-0.3, 0.1),
        alpha_step: float = 0.01,
    ) -> float:
        """Find the α minimising σ(PW) over ``subset_mask`` (Eqs. 16–17).

        Use the result with ``retrieve(toa, alpha=alpha)``. The subset
        should be spectrally diverse and flat.
        """
        n = int(round((alpha_range[1] - alpha_range[0]) / alpha_step)) + 1
        best_alpha, best_sigma = alpha_range[0], np.inf
        for alpha in np.linspace(*alpha_range, n):
            pw = self.retrieve(toa_reflectance, pw_initial, n_iter, float(alpha), subset_mask)
            sigma = float(pw.std())
            if sigma < best_sigma:
                best_alpha, best_sigma = float(alpha), sigma
        log.debug(f"optimize_path_radiance: alpha={best_alpha:.4f} (σ={best_sigma:.4g})")
        return best_alpha
