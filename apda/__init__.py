"""APDA — Atmospheric Precorrected Differential Absorption.

Retrieves columnar water vapour from hyperspectral imagery using a
pre-computed radiative transfer look-up table (LUT).

Algorithm: Schläpfer et al. 1998, RSE 65:353-366.
"""

from apda.apda import APDA
from apda.bands import ChannelSet, select_channels
from apda.conversion import convert_ratio_to_pw, fit_exponential
from apda.lut import APDALUT
from apda.ratio import compute_apda_ratio
from apda.scene import retrieve, retrieve_scene, scene_geometry

__all__ = [
    "APDA",
    "APDALUT",
    "ChannelSet",
    "compute_apda_ratio",
    "convert_ratio_to_pw",
    "fit_exponential",
    "retrieve",
    "retrieve_scene",
    "scene_geometry",
    "select_channels",
]
