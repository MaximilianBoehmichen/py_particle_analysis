import re
from pathlib import Path

import numpy as np
import pytest

from pypana.data.defs import Quantity
from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.tsi.tsi_cpc3775 import (
    TSICPC3775ColumnInstrumentReader,
    TSICPC3775InstrumentReader,
)
from readers.defs import (
    TSICPC3775_COLUMN,
    TSICPC3775_FILE1,
    TSICPC3775_FILE2,
    TSICPC3775_FILE3,
    TSICPC3775_FILE4,
)

MEASUREMENTS_FILE1 = 86
MEASUREMENTS_FILE2 = 17
MEASUREMENTS_FILE3 = 12
MEASUREMENTS_FILE4 = 52
SAMPLES_PER_MEASUREMENT_FILE1 = 60
TOTAL_SAMPLES_FILE3 = 4804
LENGTHS_FILE3_START = (544, 544, 476, 360)
INTERVAL_COLUMN = 4
TIMESTAMP_COLUMNS_PER_SAMPLE = 5
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
COLUMN_HEAD_LINES = 23
FIRST_DATA_LINE = 18


class TestTSICPC3775InstrumentReader:
    """Test suite for the TSICPC3775InstrumentReader class."""

    def test_read_file1(self) -> None:
        """Reads all samples from the fixture without count columns."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE1).read()
        assert len(data.measurements) == MEASUREMENTS_FILE1

    def test_read_file2(self) -> None:
        """Reads all samples from the fixture with interleaved count columns."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE2).read()
        assert len(data.measurements) == MEASUREMENTS_FILE2

    def test_is_time_series_only(self) -> None:
        """A CPC sample carries a Number TimeSeries and no size distribution."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE1).read()
        m = data.measurements[0]

        assert m.axis is None
        assert not m.distributions
        assert Quantity.NUMBER in m.series
        assert m.series[Quantity.NUMBER].n_samples == SAMPLES_PER_MEASUREMENT_FILE1

    def test_timestamps_follow_averaging_interval(self) -> None:
        """Each value is stamped at the end of its interval, one interval after the start time."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE1).read()
        m = data.measurements[0]
        series = m.series[Quantity.NUMBER]

        assert series.timestamps[0] == np.datetime64(m.time, "ms") + np.timedelta64(
            1, "s"
        )
        assert np.all(np.diff(series.timestamps) == np.timedelta64(1, "s"))

    def test_instrument_errors_none_is_kept(self) -> None:
        """The text 'None' in the Instrument Errors column is not read as missing."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE1).read()

        assert all(
            m.other["instrument_errors"] == "None" for m in data.measurements.values()
        )

    def test_unsupported_interval_raises(self, tmp_path: Path) -> None:
        """An averaging interval other than 1 s raises."""
        lines = TSICPC3775_FILE1.read_text(encoding="iso-8859-1").splitlines()
        header_pos = next(
            i for i, line in enumerate(lines) if line.startswith("Sample #")
        )

        for i in range(header_pos + 1, len(lines)):
            fields = lines[i].split("\t")
            fields[INTERVAL_COLUMN] = "2"
            lines[i] = "\t".join(fields)

        path = tmp_path / "interval.txt"
        path.write_text("\n".join(lines), encoding="iso-8859-1")

        with pytest.raises(ReaderNotImplementedError):
            TSICPC3775InstrumentReader(path).read()

    def test_variable_sample_lengths(self) -> None:
        """Rows shorter than the widest one only yield their own number of samples."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE3).read()
        assert len(data.measurements) == MEASUREMENTS_FILE3

        lengths = tuple(
            data.measurements[i].series[Quantity.NUMBER].n_samples
            for i in range(len(LENGTHS_FILE3_START))
        )
        assert lengths == LENGTHS_FILE3_START

        total = sum(
            m.series[Quantity.NUMBER].n_samples for m in data.measurements.values()
        )
        assert total == TOTAL_SAMPLES_FILE3

    def test_counts_aligned_when_present(self) -> None:
        """The raw count columns are stored aligned with the TimeSeries samples."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE2).read()
        m = data.measurements[0]

        assert len(m.other["counts"]) == m.series[Quantity.NUMBER].n_samples

    def test_counts_absent_without_count_columns(self) -> None:
        """Files exported without count columns store no counts array."""
        data = TSICPC3775InstrumentReader(TSICPC3775_FILE1).read()
        assert "counts" not in data.measurements[0].other

    def test_column_equals_row(self) -> None:
        """The column export holds the same samples as the row export."""
        row = TSICPC3775InstrumentReader(TSICPC3775_FILE4).read()
        column = TSICPC3775ColumnInstrumentReader(TSICPC3775_COLUMN).read()

        assert column.measurements.keys() == row.measurements.keys()
        assert len(column.measurements) == MEASUREMENTS_FILE4

        for i, m in row.measurements.items():
            other = column.measurements[i]
            series = m.series[Quantity.NUMBER]
            other_series = other.series[Quantity.NUMBER]

            assert other.time == m.time
            assert np.array_equal(other_series.timestamps, series.timestamps)
            assert np.array_equal(other_series.values, series.values)
            assert len(other.other["counts"]) == series.n_samples

            for key, value in m.other.items():
                assert other.other[key] == pytest.approx(value)

    def test_row_and_column_exports_are_told_apart(self) -> None:
        """Each reader rejects the other layout without an error."""
        assert TSICPC3775ColumnInstrumentReader.can_read(TSICPC3775_FILE4) is False
        assert TSICPC3775InstrumentReader.can_read(TSICPC3775_COLUMN) is False

    @pytest.mark.parametrize(
        ("line", "old", "new", "error"),
        [
            (7, "1.0", "2.0", ReaderNotImplementedError),
            (FIRST_DATA_LINE, "\t0\t0\t", "\t0\t1\t", ReaderNotImplementedError),
            (FIRST_DATA_LINE, "\t945.4", "\tx945.4", ReadError),
        ],
    )
    def test_edited_column_export_raises(
        self, tmp_path: Path, line: int, old: str, new: str, error: type[ReadError]
    ) -> None:
        """An unsupported interval, non-zero analog inputs and a broken value raise."""
        with Path.open(TSICPC3775_COLUMN, encoding="iso-8859-1") as f:
            lines = [next(f) for _ in range(COLUMN_HEAD_LINES)]

        assert old in lines[line]
        lines[line] = lines[line].replace(old, new, 1)

        file_path = tmp_path / "edited.txt"
        file_path.write_text("".join(lines), encoding="iso-8859-1")

        with pytest.raises(error):
            TSICPC3775ColumnInstrumentReader(file_path).read()
