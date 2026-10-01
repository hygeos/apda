"""Measurement/reference channel sets for the H₂O absorption bands.

Wavelengths are in µm (the HYPS LUT unit).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSet:
    """Measurement and reference wavelengths for one absorption band.

    ``meas`` lie inside the absorption feature, ``ref`` on the continuum
    either side of it. Pass the same object to :meth:`APDALUT.collapse` and
    :class:`APDA` so both use identical channels.
    """

    band: str
    meas: tuple[float, ...]
    ref: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "meas", tuple(map(float, self.meas)))
        object.__setattr__(self, "ref", tuple(map(float, self.ref)))

    @property
    def all(self) -> tuple[float, ...]:
        """All wavelengths (meas then ref)."""
        return self.meas + self.ref


#: Curated channel sets per band (chosen by inspection, validated on HYPS).
CURATED_BANDS: dict[str, ChannelSet] = {
    "910": ChannelSet(
        "910",
        meas=(0.90, 0.91, 0.92),
        ref=(0.85, 0.86, 0.87, 0.88, 0.98, 0.99, 1.00),
    ),
    "1100": ChannelSet(
        "1100",
        meas=(1.09, 1.10, 1.11),
        ref=(1.04, 1.05, 1.06, 1.20, 1.21, 1.22),
    ),
}


def select_channels(band: str) -> ChannelSet:
    """Return the curated channel set for ``band`` (``"910"`` or ``"1100"``)."""
    try:
        return CURATED_BANDS[band]
    except KeyError:
        raise ValueError(
            f"No channel set for band '{band}'. Available: {sorted(CURATED_BANDS)}."
        ) from None
