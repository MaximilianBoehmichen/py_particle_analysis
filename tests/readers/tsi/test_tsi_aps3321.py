from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypana.config import UnitScale
from pypana.data.defs import Quantity
from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.tsi.tsi_aps3321 import (
    TSIAPS3321ColumnInstrumentReader,
    TSIAPS3321InstrumentReader,
)
from readers.defs import (
    TSIAPS3321_COLUMN,
    TSIAPS3321_FILE1,
    TSIAPS3321_FILE2,
    TSIAPS3321_RAW_ROWS,
    TSIAPS3321_ROW_MASS,
    TSIAPS3321_ROW_NUMBER,
    TSIAPS3321_ROW_SURFACE,
    TSIAPS3321_ROW_TAB,
)

MEASUREMENTS_FILE1 = 32
MEASUREMENTS_FILE2 = 107
BINS = 52
CHANNELS_PER_DECADE = 32
FIRST_BIN_DECADES = 0.25
TOTAL_CONC_COLUMN = "Total Conc."
FIRST_BIN_WEIGHTING_DIAMETER = 0.3 * UnitScale.MICRO
HEADER_POSITION = 6


def _read_table(path: Path) -> pd.DataFrame:
    header_pos = next(
        i
        for i, line in enumerate(path.read_text(encoding="iso-8859-1").splitlines())
        if line.startswith("Sample #")
    )

    return pd.read_table(path, header=header_pos, encoding="iso-8859-1")


class TestTSIAPS3321InstrumentReader:
    """Test suite for the TSIAPS3321InstrumentReader class."""

    def test_first_bin_keeps_measured_value(self) -> None:
        """The first bin keeps its measured dN/dlogdp and spans 0.25 decades."""
        for file_path in (TSIAPS3321_FILE1, TSIAPS3321_FILE2):
            data = TSIAPS3321InstrumentReader(file_path).read()
            table = _read_table(file_path)
            first_bin = table.iloc[:, 4].to_numpy(dtype=float)

            for i, expected in enumerate(first_bin):
                m = data.measurements[i]
                assert np.isclose(m["dN/dlogdp"][0], expected)
                assert np.isclose(m.delta_log_d_p[0], FIRST_BIN_DECADES)

    def test_total_matches_reported(self) -> None:
        """The integrated total matches the instrument's own reported Total Conc."""
        for file_path in (TSIAPS3321_FILE1, TSIAPS3321_FILE2):
            data = TSIAPS3321InstrumentReader(file_path).read()
            table = _read_table(file_path)
            reported = (
                table[TOTAL_CONC_COLUMN]
                .str.extract(r"^([\d.eE+-]+)")[0]
                .to_numpy(dtype=float)
            )
            totals = np.array(
                [data.measurements[i].total for i in range(len(reported))]
            )

            assert np.allclose(totals, reported, rtol=1e-4, atol=1e-6)

    def test_read_file1(self) -> None:
        """Reads all scans from the first APS 3321 fixture."""
        data = TSIAPS3321InstrumentReader(TSIAPS3321_FILE1).read()
        assert len(data.measurements) == MEASUREMENTS_FILE1

    def test_read_file2(self) -> None:
        """Reads all scans from the second APS 3321 fixture."""
        TSIAPS3321InstrumentReader(TSIAPS3321_FILE2).read()

    def test_measurement_fields(self) -> None:
        """A parsed scan stores an aerodynamic dN/dlogdp distribution on an exact grid."""
        data = TSIAPS3321InstrumentReader(TSIAPS3321_FILE1).read()
        m = data.measurements[0]

        assert Quantity.NUMBER in m
        assert m.n_bins == BINS
        assert m.diameter_type == "aerodynamic"
        assert len(m["dN/dlogdp"]) == BINS
        assert np.allclose(m.delta_log_d_p[1:], 1 / CHANNELS_PER_DECADE)
        assert np.all(np.diff(m.bin_boundaries) > 0)

    def test_delimiter_variants_are_identical(self, tmp_path: Path) -> None:
        """Comma and semicolon exports hold the same data as the tab export."""
        text = TSIAPS3321_ROW_TAB.read_text(encoding="iso-8859-1")
        tab = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_TAB).read()

        for delimiter in (",", ";"):
            file_path = tmp_path / "delimited.TXT"
            file_path.write_text(text.replace("\t", delimiter), encoding="iso-8859-1")

            data = TSIAPS3321InstrumentReader(file_path).read()

            assert data.measurements.keys() == tab.measurements.keys()
            assert data.other_info == tab.other_info

            for i, m in tab.measurements.items():
                other = data.measurements[i]

                assert other.time == m.time
                assert np.array_equal(other["dN/dlogdp"], m["dN/dlogdp"])
                assert np.array_equal(other.bin_boundaries, m.bin_boundaries)

    @pytest.mark.parametrize("file_path", TSIAPS3321_RAW_ROWS)
    def test_raw_counts_raise(self, file_path: Path) -> None:
        """Raw count exports in every delimiter raise."""
        with pytest.raises(ReaderNotImplementedError):
            TSIAPS3321InstrumentReader(file_path).read()

    def test_column_export_is_not_a_row_file(self) -> None:
        """A column oriented export is rejected without an error."""
        assert TSIAPS3321InstrumentReader.can_read(TSIAPS3321_COLUMN) is False

    def test_number_per_bin_equals_number_per_decade(self) -> None:
        """The per bin export divided by the bin widths equals the dN/dlogDp export."""
        per_decade = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_TAB).read()
        per_bin = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_NUMBER).read()

        assert per_bin.measurements.keys() == per_decade.measurements.keys()

        for i, m in per_decade.measurements.items():
            assert np.allclose(
                per_bin.measurements[i]["dN/dlogdp"], m["dN/dlogdp"], rtol=1e-4
            )

    def test_surface_total_matches_reported(self) -> None:
        """The surface total equals the reported total converted from um2 to m2."""
        data = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_SURFACE).read()
        table = _read_table(TSIAPS3321_ROW_SURFACE)
        reported = (
            table[TOTAL_CONC_COLUMN]
            .str.extract(r"^([\d.eE+-]+)")[0]
            .to_numpy(dtype=float)
        )

        for i, expected in enumerate(reported):
            m = data.measurements[i]

            assert m.quantities == (Quantity.SURFACE,)
            assert np.isclose(
                m.total, expected * UnitScale.MICRO**2, rtol=1e-5, atol=1e-30
            )

    def test_surface_follows_number(self) -> None:
        """dS equals the number weighted with the bin midpoint, and 0.3 um in the first bin."""
        number = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_TAB).read()
        surface = TSIAPS3321InstrumentReader(TSIAPS3321_ROW_SURFACE).read()

        for i, n in number.measurements.items():
            d_n = n["dN/dlogdp"]
            d_s = surface.measurements[i]["dS/dlogdp"]
            filled = d_n > 0

            assert np.allclose(
                d_s[1:][filled[1:]],
                np.pi * n.d_p[1:][filled[1:]] ** 2 * d_n[1:][filled[1:]],
                rtol=1e-4,
            )

            if filled[0]:
                assert np.isclose(
                    d_s[0],
                    np.pi * FIRST_BIN_WEIGHTING_DIAMETER**2 * d_n[0],
                    rtol=1e-4,
                )

    def test_mass_raises(self) -> None:
        """Mass exports raise."""
        with pytest.raises(ReaderNotImplementedError):
            TSIAPS3321InstrumentReader(TSIAPS3321_ROW_MASS).read()

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("(#/cm\u00b3)", "(mg/m\u00b3)"),
            ("\tdN/dlogDp\t", "\tdN\t"),
        ],
    )
    def test_inconsistent_data_type_raises(
        self, tmp_path: Path, old: str, new: str
    ) -> None:
        """A wrong total unit or mixed data types raise."""
        text = TSIAPS3321_ROW_TAB.read_text(encoding="iso-8859-1")
        assert old in text

        file_path = tmp_path / "edited.TXT"
        file_path.write_text(text.replace(old, new, 1), encoding="iso-8859-1")

        with pytest.raises(ReadError):
            TSIAPS3321InstrumentReader(file_path).read()

    def test_column_table_equals_row_table(self) -> None:
        """The transposed column export holds the same table as the row export."""
        row = TSIAPS3321InstrumentReader(TSIAPS3321_RAW_ROWS[0])
        column = TSIAPS3321ColumnInstrumentReader(TSIAPS3321_COLUMN)

        pd.testing.assert_frame_equal(
            column._read_table("\t", HEADER_POSITION),
            row._read_table("\t", HEADER_POSITION),
        )

    def test_row_export_is_not_a_column_file(self) -> None:
        """A row oriented export is rejected by the column reader without an error."""
        assert TSIAPS3321ColumnInstrumentReader.can_read(TSIAPS3321_ROW_TAB) is False

    def test_column_raw_counts_raise(self) -> None:
        """The column export holds raw counts and raises."""
        with pytest.raises(ReaderNotImplementedError):
            TSIAPS3321ColumnInstrumentReader(TSIAPS3321_COLUMN).read()

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("Start Time\t11:48:06\t", "Start Time\t"),
            ("Sample #\t1\t2\t", "Sample #\t2\t1\t"),
        ],
    )
    def test_edited_column_export_raises(
        self, tmp_path: Path, old: str, new: str
    ) -> None:
        """A missing value or unordered sample numbers raise."""
        text = TSIAPS3321_COLUMN.read_text(encoding="iso-8859-1")
        assert old in text

        file_path = tmp_path / "edited.TXT"
        file_path.write_text(text.replace(old, new, 1), encoding="iso-8859-1")

        with pytest.raises(ReadError):
            TSIAPS3321ColumnInstrumentReader(file_path).read()

    @pytest.mark.parametrize(
        ("old", "new", "error"),
        [
            (
                "Stokes Correction\toff",
                "Stokes Correction\ton",
                ReaderNotImplementedError,
            ),
            (
                "Stokes Correction\toff",
                "Stokes Correction\tmaybe",
                ReaderNotImplementedError,
            ),
            ("Density\t1\n", "Density\t2\n", ReaderNotImplementedError),
            ("Density\t1\n", "Density\t1.5\n", ReaderNotImplementedError),
            ("Density\t1\n", "", ReadError),
            ("Density\t1\n", "Density\tx\n", ReadError),
        ],
    )
    def test_unsupported_settings_raise(
        self, tmp_path: Path, old: str, new: str, error: type[ReadError]
    ) -> None:
        """Stokes correction other than off and a density other than 1 raise."""
        text = TSIAPS3321_ROW_TAB.read_text(encoding="iso-8859-1")
        assert old in text

        file_path = tmp_path / "edited.TXT"
        file_path.write_text(text.replace(old, new, 1), encoding="iso-8859-1")

        with pytest.raises(error):
            TSIAPS3321InstrumentReader(file_path).read()
