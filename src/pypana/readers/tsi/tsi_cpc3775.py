"""Implementation of a Reader for the TSI Condensation Particle Counter 3775.

This module provides the corresponding reader for the produced files of a TSI CPC 3775.

References:
    https://tsi.com/discontinued-products/condensation-particle-counter-3775
"""

import itertools
import re
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pypana.data.defs import FloatArray, Quantity
from pypana.data.instrument_data import InstrumentData
from pypana.data.measurement import Measurement
from pypana.data.time_series import TimeSeries
from pypana.readers.base_instrument_reader import BaseInstrumentReader
from pypana.readers.base_reader import InputType
from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.tsi.utils import is_basic_tsi_format_file, read_metadata_block


class TSICPC3775InstrumentReader(BaseInstrumentReader):
    """Instrument reader for TSI CPC 3775."""

    _device_name = "TSI CPC 3775"
    _encoding = "iso-8859-1"
    _input_type = InputType.FILE
    _SUPPORTED_INTERVAL = 1

    _HEADER_START = (
        "Sample #\tStart Date\tStart Time\tSample Length\tAveraging Interval (s)"
    )
    _SAMPLES_AS_ROWS = True
    _DATETIME_FORMAT = "%m/%d/%y %H:%M:%S"

    _CONC_COLUMN_PATTERN = re.compile(r"\[\d+(\.\d+)?\]\s+Conc")
    _COUNT_COLUMN_PATTERN = re.compile(r"\[\d+(\.\d+)?\]\s+Count")

    _OTHER_COLUMNS = {
        "instrument_id": "Instrument ID",
        "instrument_errors": "Instrument Errors",
        "conc_mean": "Conc Mean",
        "conc_min": "Conc Min",
        "conc_max": "Conc Max",
        "conc_std_dev": "Conc Std Dev",
    }

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Checks whether a given path may include a TSI CPC 3775 output file that can be read.

        Args:
            path: The path to the input file.

        Returns:
            Whether the read test succeeded when applying the TSI CPC 3775 format.

        Raises:
            ReadError: If confident enough that the input is from TSI CPC 3775, but the data suggests otherwise.
                This may happen because the input files were manually edited in unsafe places or this package
                does not yet fully implement this device's formats.
                Note: the absence of ReadError in this method does not guarantee the input is parseable.
        """
        anchors = ["Sample File", "Model\t3775", cls._HEADER_START]

        return is_basic_tsi_format_file(
            path,
            anchors,
            check_sample_order=cls._SAMPLES_AS_ROWS,
            encoding=cls._encoding,
        )

    def read(self) -> InstrumentData:
        """Read the given file and convert its data into the pypana format.

        Returns:
            InstrumentData: The pypana instrument on which further analysis can be conducted.

        Raises:
            ReadError: If an error occurs while reading the file.
        """
        other_info: dict[Hashable, Any] = {}
        header_line = 0

        try:
            with Path.open(self._path, "r", encoding=self._encoding) as f:
                for i, line in enumerate(f):
                    if line.startswith("Sample File"):
                        other_info["sample_file"] = line[12:].strip()
                    elif line.startswith("Model"):
                        other_info["model"] = line[6:].strip()
                    elif line.startswith(self._HEADER_START):
                        header_line = i
                        break

        except (FileNotFoundError, UnicodeDecodeError) as e:
            raise ReadError(f"{e}", path=self._path) from e

        if header_line == 0:
            raise ReadError(
                message="The file does not contain the start of the TSI CPC 3775 header.",
                path=self._path,
            )

        try:
            data = pd.read_table(
                self._path,
                sep="\t",
                skiprows=header_line,
                index_col=False,
                encoding=self._encoding,
                keep_default_na=False,
                na_values=[""],
            )

            conc_columns = [
                c for c in data.columns if self._CONC_COLUMN_PATTERN.match(c)
            ]
            count_columns = [
                c for c in data.columns if self._COUNT_COLUMN_PATTERN.match(c)
            ]

            if not conc_columns:
                raise ReadError(
                    message="No concentration columns found.", path=self._path
                )

        except (FileNotFoundError, ValueError) as e:
            raise ReadError(f"{e}", path=self._path) from e

        measurements: dict[int, Measurement] = {}

        for row in data.to_dict("records"):
            try:
                scan_nr, measurement = self._row_to_measurement(
                    row, conc_columns, count_columns
                )
                measurements[scan_nr - 1] = measurement

            except (ValueError, KeyError) as e:
                raise ReadError(f"{e}", path=self._path) from e

        if not measurements:
            raise ReadError(message="No valid measurements to import!", path=self._path)

        return InstrumentData(
            measurements=measurements,
            device_name=self._device_name,
            file_path=self._path,
            other_info=other_info,
        )

    def _row_to_measurement(
        self,
        row: dict[Hashable, Any],
        conc_columns: list[str],
        count_columns: list[str],
    ) -> tuple[int, Measurement]:
        """Build one Measurement from a parsed CPC sample row.

        Args:
            row: One record from the parsed dataframe.
            conc_columns: Column labels holding the per-interval concentrations.
            count_columns: Column labels holding the per-interval raw counts (may be empty).

        Returns:
            The 1-based sample number and its Measurement.
        """
        scan_nr = int(row["Sample #"])
        start = datetime.strptime(
            f"{row['Start Date']} {row['Start Time']}", self._DATETIME_FORMAT
        )
        interval = int(row["Averaging Interval (s)"])
        sample_length = int(row["Sample Length"])

        self._check_interval(interval)

        n_samples = min(sample_length // interval, len(conc_columns))

        concentrations = np.array(
            [row[c] for c in conc_columns[:n_samples]], dtype=float
        )
        counts = (
            np.array([row[c] for c in count_columns[:n_samples]], dtype=float)
            if count_columns
            else None
        )
        statistics: dict[Hashable, Any] = {
            key: row[column] for key, column in self._OTHER_COLUMNS.items()
        }

        measurement = self._build_measurement(
            scan_nr,
            start,
            interval,
            sample_length,
            "" if pd.isna(row["Title"]) else str(row["Title"]),
            concentrations,
            counts,
            statistics,
        )

        return scan_nr, measurement

    def _check_interval(self, interval: int) -> None:
        """Checks that the averaging interval is supported.

        Args:
            interval: The averaging interval in seconds.

        Raises:
            ReaderNotImplementedError: If the averaging interval is not supported.
        """
        if interval != self._SUPPORTED_INTERVAL:
            raise ReaderNotImplementedError(
                f"Only an averaging interval of {self._SUPPORTED_INTERVAL} s is implemented "
                f"for TSI CPC 3775 (got {interval} s)."
            )

    def _build_measurement(
        self,
        scan_nr: int,
        start: datetime,
        interval: int,
        sample_length: int,
        title: str,
        concentrations: FloatArray,
        counts: FloatArray | None,
        statistics: dict[Hashable, Any],
    ) -> Measurement:
        """Build one Measurement from the values of a CPC sample.

        Args:
            scan_nr: The 1-based sample number.
            start: The start of the sample.
            interval: The averaging interval in seconds.
            sample_length: The sample length in seconds.
            title: The title of the sample.
            concentrations: The concentration of each interval.
            counts: The raw counts of each interval, if reported.
            statistics: The other values reported for the sample, by name.

        Returns:
            The Measurement of the sample.
        """
        # TSI stamps each value at the end of its averaging interval
        timestamps = np.datetime64(start, "ms") + np.arange(
            1, len(concentrations) + 1
        ) * np.timedelta64(interval, "s")

        number = TimeSeries(
            quantity=Quantity.NUMBER,
            timestamps=timestamps,
            values=concentrations,
        )

        other: dict[Hashable, Any] = {
            "title": title,
            "sample_length": sample_length,
            "averaging_interval": interval,
            **statistics,
        }

        if counts is not None:
            other["counts"] = counts

        return Measurement(
            scan_nr=scan_nr,
            time=start,
            series={Quantity.NUMBER: number},
            other=other,
        )


class TSICPC3775ColumnInstrumentReader(TSICPC3775InstrumentReader):
    """Instrument reader for TSI CPC 3775 files with one block of columns per sample."""

    _HEADER_START = "Sample #\t1\t\t\t"
    _SAMPLES_AS_ROWS = False
    _BLOCK_WIDTH = 4
    _FINISHED_BLOCK_WIDTH = 3
    _ELAPSED_LABEL = "Elapsed (s)"
    _STATISTIC_LABELS = {
        "conc_mean": "Mean",
        "conc_min": "Min",
        "conc_max": "Max",
        "conc_std_dev": "Std. Dev.",
    }

    def read(self) -> InstrumentData:
        """Read the given file and convert its data into the pypana format.

        Each sample has a block of four columns: concentration, count and two analog inputs.
        A finished sample leaves three empty cells instead of four.

        Returns:
            InstrumentData: The pypana instrument on which further analysis can be conducted.

        Raises:
            ReadError: If an error occurs while reading the file.
            ReaderNotImplementedError: If the interval or the analog inputs are not supported.
        """
        try:
            header_pos, metadata = read_metadata_block(
                self._path, self._HEADER_START, "\t", encoding=self._encoding
            )

            with Path.open(self._path, "r", encoding=self._encoding) as f:
                rows = [
                    line.rstrip("\r\n").split("\t")
                    for line in itertools.islice(f, header_pos, None)
                ]

            measurements = self._read_samples(rows)

        except (FileNotFoundError, UnicodeDecodeError) as e:
            raise ReadError(f"{e}", path=self._path) from e
        except (ValueError, IndexError, KeyError) as e:
            raise ReadError(f"{e}", path=self._path) from e

        if not measurements:
            raise ReadError(message="No valid measurements to import!", path=self._path)

        return InstrumentData(
            measurements=measurements,
            device_name=self._device_name,
            file_path=self._path,
            other_info={
                "sample_file": metadata["Sample File"],
                "model": metadata["Model"],
            },
        )

    def _read_samples(self, rows: list[list[str]]) -> dict[int, Measurement]:
        """Decodes the samples from the table rows, starting at the line with the sample numbers.

        Args:
            rows: The split lines of the table.

        Returns:
            The measurements by 0-based sample index.

        Raises:
            ReaderNotImplementedError: If the interval or the analog inputs are not supported.
        """
        sample_ids = [int(cell) for cell in rows[0][1:] if cell]
        n_samples = len(sample_ids)

        elapsed_pos = next(
            (i for i, row in enumerate(rows) if row[0] == self._ELAPSED_LABEL), None
        )

        if elapsed_pos is None:
            raise ReadError(
                message=f"The file does not contain a line starting with '{self._ELAPSED_LABEL}'.",
                path=self._path,
            )

        values = {
            row[0]: [row[1 + self._BLOCK_WIDTH * j] for j in range(n_samples)]
            for row in rows[1:elapsed_pos]
            if row[0]
        }

        lengths = [
            sum(
                int(part) * factor
                for part, factor in zip(length.split(":"), (3600, 60, 1), strict=True)
            )
            for length in values["Sample Length"]
        ]

        concentrations: list[list[float]] = [[] for _ in range(n_samples)]
        counts: list[list[float]] = [[] for _ in range(n_samples)]
        analog_sum = 0.0

        for row in rows[elapsed_pos + 1 :]:
            if len(row) == 1:
                break

            elapsed = float(row[0])
            position = 1

            for j in range(n_samples):
                if elapsed <= lengths[j]:
                    block = row[position : position + self._BLOCK_WIDTH]
                    concentrations[j].append(float(block[0]))
                    counts[j].append(float(block[1]))
                    analog_sum += abs(float(block[2])) + abs(float(block[3]))
                    position += self._BLOCK_WIDTH
                else:
                    position += self._FINISHED_BLOCK_WIDTH

        if analog_sum != 0:
            raise ReaderNotImplementedError(
                "Non zero analog inputs are not implemented for TSI CPC 3775."
            )

        measurements: dict[int, Measurement] = {}

        for j, scan_nr in enumerate(sample_ids):
            interval = int(float(values["Averaging Interval (secs)"][j]))
            self._check_interval(interval)

            measurements[scan_nr - 1] = self._build_measurement(
                scan_nr,
                datetime.strptime(
                    f"{values['Start Date'][j]} {values['Start Time'][j]}",
                    self._DATETIME_FORMAT,
                ),
                interval,
                lengths[j],
                values["Title"][j],
                np.array(concentrations[j], dtype=float),
                np.array(counts[j], dtype=float),
                {
                    "instrument_id": values["Instrument ID"][j],
                    "instrument_errors": values["Instrument Errors"][j],
                    **{
                        key: float(values[label][j])
                        for key, label in self._STATISTIC_LABELS.items()
                    },
                },
            )

        return measurements
