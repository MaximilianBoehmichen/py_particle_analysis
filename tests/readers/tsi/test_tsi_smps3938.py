from pathlib import Path

import numpy as np
import pytest

from pypana.data.defs import Quantity
from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.tsi.tsi_smps3938 import (
    TSISMPS3938ColumnInstrumentReader,
    TSISMPS3938InstrumentReader,
)
from readers.defs import (
    TSISMPS3938_COLUMN,
    TSISMPS3938_COLUMN_ALL,
    TSISMPS3938_COLUMN_RAW,
    TSISMPS3938_FILE3,
    TSISMPS3938_FILE4,
    TSISMPS3938_FILE5,
    TSISMPS3938_ROW_ALL,
    TSISMPS3938_ROW_COMMA,
    TSISMPS3938_ROW_SEMICOLON,
    TSISMPS3938_ROW_SURFACE,
    TSISMPS3938_ROW_TAB,
    TSISMPS3938_ROW_VOLUME,
    TSISMPS3938_UNSUPPORTED_WEIGHTS,
)

CHANNELS_PER_DECADE = 64
TOTAL_CONC_COLUMN = "Total Conc. (#/cm\u00b3)"
SURFACE_COLUMN = "Total Conc. (nm\u00b2/cm\u00b3)"
VOLUME_COLUMN = "Total Conc. (nm\u00b3/cm\u00b3)"
NM = 1e-9
MIDPOINT_RTOL = 1e-3
CORRECTION_FLAGS = (
    "multiple_charge_correction",
    "nanoparticle_agglomerate_mobility_analysis",
    "diffusion_correction",
)


class TestTSISMPS3938InstrumentReader:
    """Test suite for the native TSISMPS3938InstrumentReader."""

    def test_read_counts(self) -> None:
        """Each native fixture yields one Measurement per row."""
        for file_path, expected in (
            (TSISMPS3938_FILE3, 6),
            (TSISMPS3938_FILE4, 23),
            (TSISMPS3938_FILE5, 14),
        ):
            data = TSISMPS3938InstrumentReader(file_path).read()
            assert len(data.measurements) == expected

        """A scan stores a mobility dN/dlogdp distribution on an exact 64/decade grid."""
        data = TSISMPS3938InstrumentReader(TSISMPS3938_FILE5).read()
        m = data.measurements[0]

        assert Quantity.NUMBER in m
        assert m.diameter_type == "mobility"
        assert np.allclose(m.delta_log_d_p, 1 / CHANNELS_PER_DECADE)

    def test_total_matches_reported(self) -> None:
        """The integrated total matches the instrument's own reported Total Conc."""
        data = TSISMPS3938InstrumentReader(TSISMPS3938_FILE5).read()
        m = max(data.measurements.values(), key=lambda x: x.total)

        reported = float(m.other[TOTAL_CONC_COLUMN])
        assert np.isclose(m.total, reported, rtol=1e-3)

    def test_delimiter_variants_are_identical(self) -> None:
        """Comma and semicolon exports hold the same data as the tab export."""
        tab = TSISMPS3938InstrumentReader(TSISMPS3938_ROW_TAB).read()

        for file_path in (TSISMPS3938_ROW_COMMA, TSISMPS3938_ROW_SEMICOLON):
            data = TSISMPS3938InstrumentReader(file_path).read()

            assert data.measurements.keys() == tab.measurements.keys()

            for i, m in tab.measurements.items():
                other = data.measurements[i]

                assert other.time == m.time
                assert np.array_equal(other["dN/dlogdp"], m["dN/dlogdp"])
                assert np.array_equal(other.bin_boundaries, m.bin_boundaries)

    def test_column_export_is_not_a_row_file(self) -> None:
        """A column oriented export is rejected without an error."""
        assert TSISMPS3938InstrumentReader.can_read(TSISMPS3938_COLUMN) is False

    def test_surface_and_volume_totals_match_reported(self) -> None:
        """The stored totals equal the reported totals converted from nm to m."""
        for file_path, quantity, column in (
            (TSISMPS3938_ROW_SURFACE, Quantity.SURFACE, SURFACE_COLUMN),
            (TSISMPS3938_ROW_VOLUME, Quantity.VOLUME, VOLUME_COLUMN),
        ):
            data = TSISMPS3938InstrumentReader(file_path).read()

            for m in data.measurements.values():
                assert m.quantities == (quantity,)

                reported = float(m.other[column]) * NM**quantity.moment
                assert np.isclose(m.total, reported, rtol=1e-6)

    def test_surface_and_volume_follow_number(self) -> None:
        """dS and dV equal the number distribution weighted with the bin midpoint diameter."""
        number = TSISMPS3938InstrumentReader(TSISMPS3938_ROW_TAB).read()
        surface = TSISMPS3938InstrumentReader(TSISMPS3938_ROW_SURFACE).read()
        volume = TSISMPS3938InstrumentReader(TSISMPS3938_ROW_VOLUME).read()

        for i, n in number.measurements.items():
            d_p = n.d_p
            d_n = n["dN/dlogdp"]
            filled = d_n > 0

            assert np.allclose(
                surface.measurements[i]["dS/dlogdp"][filled],
                np.pi * d_p[filled] ** 2 * d_n[filled],
                rtol=MIDPOINT_RTOL,
            )
            assert np.allclose(
                volume.measurements[i]["dV/dlogdp"][filled],
                np.pi / 6 * d_p[filled] ** 3 * d_n[filled],
                rtol=MIDPOINT_RTOL,
            )

    @pytest.mark.parametrize("file_path", TSISMPS3938_UNSUPPORTED_WEIGHTS)
    def test_unsupported_weights_raise(self, file_path: Path) -> None:
        """Diameter, mass, per bin and raw count exports raise."""
        with pytest.raises(ReaderNotImplementedError):
            TSISMPS3938InstrumentReader(file_path).read()

    def test_column_equals_row(self) -> None:
        """Column exports hold the same data as the row exports."""
        for row_file, column_file in (
            (TSISMPS3938_ROW_TAB, TSISMPS3938_COLUMN),
            (TSISMPS3938_ROW_ALL, TSISMPS3938_COLUMN_ALL),
        ):
            row = TSISMPS3938InstrumentReader(row_file).read()
            column = TSISMPS3938ColumnInstrumentReader(column_file).read()

            assert column.measurements.keys() == row.measurements.keys()

            for i, m in row.measurements.items():
                other = column.measurements[i]

                assert other.time == m.time
                assert np.array_equal(
                    other["dN/dlogdp"], m["dN/dlogdp"], equal_nan=True
                )
                assert np.array_equal(other.bin_boundaries, m.bin_boundaries)

    def test_row_export_is_not_a_column_file(self) -> None:
        """A row oriented export is rejected by the column reader without an error."""
        assert TSISMPS3938ColumnInstrumentReader.can_read(TSISMPS3938_ROW_TAB) is False

    @pytest.mark.parametrize("file_path", TSISMPS3938_COLUMN_RAW)
    def test_column_raw_data_raises(self, file_path: Path) -> None:
        """Column exports with more than one column per sample raise."""
        with pytest.raises(ReaderNotImplementedError):
            TSISMPS3938ColumnInstrumentReader(file_path).read()

    def test_bin_boundaries_are_decade_aligned(self) -> None:
        """Boundaries lie exactly on 10 ** (j / channels per decade) nanometers."""
        for file_path in (TSISMPS3938_ROW_TAB, TSISMPS3938_ROW_ALL, TSISMPS3938_FILE4):
            data = TSISMPS3938InstrumentReader(file_path).read()
            m = next(iter(data.measurements.values()))

            j = np.log10(m.bin_boundaries / NM) * CHANNELS_PER_DECADE

            assert np.allclose(j, np.round(j), rtol=0, atol=1e-9)

    def test_correction_flags_are_stored(self) -> None:
        """The correction flags of the metadata block are stored as booleans."""
        plain = TSISMPS3938InstrumentReader(TSISMPS3938_ROW_TAB).read()
        corrected = TSISMPS3938InstrumentReader(TSISMPS3938_FILE4).read()

        for name in CORRECTION_FLAGS:
            assert plain.other_info[name] is False

        assert corrected.other_info["diffusion_correction"] is True
        assert corrected.other_info["multiple_charge_correction"] is False

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("Diffusion Correction\tFALSE", "Diffusion Correction\tMAYBE"),
            ("Diffusion Correction\tFALSE\n", ""),
            ("6/20/2024", "20/6/2024"),
        ],
    )
    def test_edited_metadata_or_dates_raise(
        self, tmp_path: Path, old: str, new: str
    ) -> None:
        """Unknown flag values, missing flags and day first dates raise."""
        text = TSISMPS3938_ROW_TAB.read_text(encoding="iso-8859-1")
        assert old in text

        file_path = tmp_path / "edited.TXT"
        file_path.write_text(text.replace(old, new), encoding="iso-8859-1")

        with pytest.raises(ReadError):
            TSISMPS3938InstrumentReader(file_path).read()

    def test_midpoint_off_the_grid_raises(self, tmp_path: Path) -> None:
        """A header midpoint that is not on the decade aligned grid raises."""
        lines = TSISMPS3938_ROW_TAB.read_text(encoding="iso-8859-1").split("\n")
        header = next(i for i, line in enumerate(lines) if line.startswith("Sample #"))

        fields = lines[header].split("\t")
        position = fields.index("Diameter Midpoint (nm)") + 10
        fields[position] = str(float(fields[position]) + 1.0)
        lines[header] = "\t".join(fields)

        file_path = tmp_path / "edited.TXT"
        file_path.write_text("\n".join(lines), encoding="iso-8859-1")

        with pytest.raises(ReadError):
            TSISMPS3938InstrumentReader(file_path).read()
