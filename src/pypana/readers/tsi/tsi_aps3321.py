"""Implementation of a Reader for the TSI Aerodynamic Particle Sizer Spectrometer 3321.

This module provides the corresponding reader for the produced files of a TSI APS 3321.

References:
    https://tsi.com/products/particle-sizers/supermicron-capable-particle-sizer-spectrometers/aerodynamic-particle-sizer-aps-3321
"""

import re
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from pypana.config import UnitScale
from pypana.data.bin_axis import BinAxis
from pypana.data.defs import FloatArray, Quantity
from pypana.data.instrument_data import InstrumentData
from pypana.data.measurement import Measurement
from pypana.data.size_distribution import SizeDistribution
from pypana.readers.base_instrument_reader import BaseInstrumentReader
from pypana.readers.base_reader import InputType
from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.tsi.utils import (
    is_basic_tsi_format_file,
    read_column_lines,
    read_metadata_block,
    samples_to_table,
)
from pypana.readers.utils import detect_delimiter, other_columns_to_dict


class TSIAPS3321InstrumentReader(BaseInstrumentReader):
    """Instrument reader for TSI APS 3321."""

    _device_name = "TSI APS 3321"
    _encoding = "iso-8859-1"
    _input_type = InputType.FILE

    _TYPE_LABEL = "Aerodynamic Diameter"
    _TOTAL_CONC_LABEL = "Total Conc."
    _TOTAL_CONC_PATTERN = re.compile(r"^(?P<value>.+)\((?P<unit>.+)\)$")
    _REPORTED_TYPES = {
        "dN/dlogDp": (Quantity.NUMBER, "#/cm³", False),
        "dN": (Quantity.NUMBER, "#/cm³", True),
        "dS": (Quantity.SURFACE, "µm²/cm³", True),
    }
    _BINS = 52
    _D_P_COLUMNS_COUNT = 52
    _SIZE_SCALING_FACTOR = UnitScale.MICRO.value
    _CHANNELS_PER_DECADE = 32
    _SUPPORTED_DENSITY = 1.0
    _SUPPORTED_STOKES_CORRECTION = "off"
    _HEADER_START = "Sample #\tDate\tStart Time\tAerodynamic Diameter"
    _SAMPLES_AS_ROWS = True

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Checks whether a given path may include a TSI APS 3321 output file that can be read.

        Args:
            path: The path to the input file.

        Returns:
            Whether the read test succeeded when applying the TSI APS 3321 format.

        Raises:
            ReadError: If confident enough that the input is from TSI APS 3321, but the data suggests otherwise.
                This may happen because the input files were manually edited in unsafe places or this package
                does not yet fully implement this device's formats.
                Note: the absence of ReadError in this method does not guarantee the input is parseable.
        """
        anchors = ["Sample Time", "Density", "Stokes Correction", cls._HEADER_START]

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
            ReadError: If an error occurs while reading the file, or if the Aerodynamic Diameter is not set to dN/dlogDp.
        """
        delimiter = detect_delimiter(self._path, encoding=self._encoding)

        if delimiter is None:
            raise ReadError(
                message="Unknown delimiter.",
                path=self._path,
            )

        header_pos, metadata = read_metadata_block(
            self._path, self._HEADER_START, delimiter, encoding=self._encoding
        )

        sample_time, density = self._read_settings(metadata)

        try:
            data = self._read_table(delimiter, header_pos)

            d_p_start_loc = data.columns.get_loc("Aerodynamic Diameter")
            assert isinstance(d_p_start_loc, int)
            d_p_start_column = int(d_p_start_loc) + 1

            bin_boundaries = self._read_bin_metadata(d_p_start_column, data)
            axis = BinAxis(
                bin_boundaries=bin_boundaries.copy(),
                diameter_type="aerodynamic",
            )
            bin_columns = data.columns[
                d_p_start_column : d_p_start_column + self._D_P_COLUMNS_COUNT
            ]

        except (FileNotFoundError, ValueError) as e:
            raise ReadError(f"{e}") from e

        if len(data[self._TYPE_LABEL].unique()) != 1:
            raise ReadError(
                message="The file reports more than one data type.",
                path=self._path,
            )

        float_cols = data.select_dtypes(include=["float"]).columns
        data[float_cols] = data[float_cols].astype(float)

        other_columns = data.columns[
            d_p_start_column + self._D_P_COLUMNS_COUNT : -6
        ].to_list()

        measurements: dict[int, Measurement] = {}
        for row in data.to_dict("records"):
            try:
                key, measurement = self._row_to_measurement(
                    row, axis, bin_columns, other_columns
                )
                measurements[key] = measurement

            except (ValueError, AttributeError, KeyError) as e:
                raise ReadError(f"{e}") from e

            except ReadError as e:
                raise e

        other_info: dict[Hashable, Any] = {
            "sample_time": sample_time,
            "density": density,
            "stokes_corrected": False,  # hardcoded while stokes_corrected is unsupported
        }

        return InstrumentData(
            measurements=measurements,
            device_name=self._device_name,
            file_path=self._path,
            other_info=other_info,
        )

    def _read_settings(self, metadata: dict[str, str]) -> tuple[int, float]:
        """Reads the sample time, density and Stokes correction from the metadata.

        Args:
            metadata: The metadata block of the file.

        Returns:
            The sample time and the density.

        Raises:
            ReadError: If one of the settings is missing or invalid.
            ReaderNotImplementedError: If the density is not 1 or the Stokes correction is not off.
        """
        try:
            sample_time = int(metadata["Sample Time"])
            density = float(metadata["Density"])
            stokes_correction = metadata["Stokes Correction"]
        except (KeyError, ValueError) as e:
            raise ReadError(
                message="The file does not report valid Sample Time, Density and Stokes Correction.",
                path=self._path,
            ) from e

        if (
            density != self._SUPPORTED_DENSITY
            or stokes_correction != self._SUPPORTED_STOKES_CORRECTION
        ):
            raise ReaderNotImplementedError(
                f"Only density {self._SUPPORTED_DENSITY} and Stokes Correction "
                f"{self._SUPPORTED_STOKES_CORRECTION!r} are implemented for TSI APS 3321 "
                f"(got density={density}, Stokes Correction={stokes_correction!r})."
            )

        return sample_time, density

    def _read_table(self, delimiter: str, header_pos: int) -> pd.DataFrame:
        """Reads the data table, one row per sample.

        Args:
            delimiter: The delimiter separating the columns of the file.
            header_pos: The line index of the table header.

        Returns:
            The table with one row per sample.
        """
        return pd.read_table(
            self._path,
            sep=delimiter,
            header=header_pos,
            encoding=self._encoding,
            skip_blank_lines=True,
        )

    def _row_to_measurement(
        self,
        row: dict[Hashable, Any],
        axis: BinAxis,
        bin_columns: pd.Index,
        other_columns: list[str],
    ) -> tuple[int, Measurement]:
        """Build one Measurement from a parsed APS data row.

        Args:
            row: One record from the parsed dataframe.
            axis: The shared diameter axis.
            bin_columns: Column labels holding the per-bin values.
            other_columns: Column labels carried verbatim into ``other``.

        Returns:
            The 0-based measurement key and its Measurement.

        Raises:
            ReaderNotImplementedError: If the row reports a data type that is not implemented.
            ReadError: If the reported total concentration has an unexpected unit.
        """
        data_type = row[self._TYPE_LABEL]

        if data_type not in self._REPORTED_TYPES:
            raise ReaderNotImplementedError(
                f"Only {', '.join(self._REPORTED_TYPES)} are implemented for TSI APS 3321 (got {data_type!r})."
            )

        quantity, expected_unit, per_bin = self._REPORTED_TYPES[data_type]

        match = self._TOTAL_CONC_PATTERN.match(str(row[self._TOTAL_CONC_LABEL]))

        if match is None or match["unit"] != expected_unit:
            raise ReadError(
                message=f"The total concentration is {row[self._TOTAL_CONC_LABEL]!r}, expected unit {expected_unit!r}.",
                path=self._path,
            )

        scan_nr = int(row["Sample #"])
        time = datetime.strptime(
            f"{row['Date']} {row['Start Time']}", "%m/%d/%y %H:%M:%S"
        )
        values = np.array([row[col] for col in bin_columns], dtype=float)

        if per_bin:
            values = values / axis.delta_log_d_p

        distribution = SizeDistribution(
            quantity=quantity,
            axis=axis,
            delta_dlogdp=values * self._SIZE_SCALING_FACTOR**quantity.moment,
        )

        other_info = other_columns_to_dict(row, other_columns)
        other_info["median"] = row["Median(µm)"]
        other_info["mean"] = row["Mean(µm)"]
        other_info["mode"] = row["Mode(µm)"]
        other_info["geo_std_dev"] = row["Geo. Std. Dev."]

        measurement = Measurement(
            scan_nr=scan_nr,
            time=time,
            axis=axis,
            distributions={quantity: distribution},
            other=other_info,
        )

        return scan_nr - 1, measurement

    def _read_bin_metadata(self, d_p_start_column: int, data: DataFrame) -> FloatArray:
        """Reads the metadata arrays that are constant for all measurements.

        Args:
            d_p_start_column (int): Column index of the start position of the bins.
            data (DataFrame): The parsed APS dataframe.

        Returns:
            The ``n + 1`` bin boundaries in meters.
        """
        d_p_file = (
            np.array(
                data.columns[
                    d_p_start_column : d_p_start_column + self._D_P_COLUMNS_COUNT
                ].str.replace("<", ""),
                dtype=float,
            )
            * self._SIZE_SCALING_FACTOR
        )

        delta_log = 1 / self._CHANNELS_PER_DECADE
        relative = np.arange(self._D_P_COLUMNS_COUNT) * delta_log
        offset = float(np.mean(np.log10(d_p_file[1:]) - relative[1:]))
        d_p = 10.0 ** (relative + offset)

        half_step = 10.0 ** (delta_log / 2)
        boundaries = np.asarray(
            np.append(d_p / half_step, d_p[-1] * half_step), dtype=float
        )
        boundaries[0] = (
            10 ** (np.log10(boundaries[1]) - 0.25)
        )  # first bin is normalized over 0.25 decades, https://github.com/Haisch-Group/pypana/issues/20

        return boundaries


class TSIAPS3321ColumnInstrumentReader(TSIAPS3321InstrumentReader):
    """Instrument reader for TSI APS 3321 files with one column per sample."""

    _HEADER_START = "Sample #\t1"
    _SAMPLES_AS_ROWS = False
    _SUPPORTED_DELIMITER = "\t"
    _TOTAL_CONC_LABEL_PATTERN = re.compile(r"^Total Conc\.\((?P<unit>.+)\)$")

    def _read_table(self, delimiter: str, header_pos: int) -> pd.DataFrame:
        """Reads the data table and transposes it to one row per sample.

        The column export reports the unit of the total concentration in its label,
        the row export in every value. The unit is moved to the values.

        Args:
            delimiter: The delimiter separating the columns of the file.
            header_pos: The line index of the table header.

        Returns:
            The table with one row per sample, as in the row oriented export.

        Raises:
            ReaderNotImplementedError: If the delimiter is not a tab or there is more than one column per sample.
            ReadError: If the sample numbers, the number of values per row or the total concentration label are inconsistent.
        """
        if delimiter != self._SUPPORTED_DELIMITER:
            raise ReaderNotImplementedError(
                "Only tab delimited column exports are implemented for TSI APS 3321."
            )

        labels, samples = read_column_lines(
            self._path, header_pos, delimiter, encoding=self._encoding
        )

        matches = [
            (i, match)
            for i, label in enumerate(labels)
            if (match := self._TOTAL_CONC_LABEL_PATTERN.match(label))
        ]

        if len(matches) != 1:
            raise ReadError(
                message="The file does not report exactly one total concentration with a unit.",
                path=self._path,
            )

        index, match = matches[0]
        labels[index] = self._TOTAL_CONC_LABEL

        for sample in samples:
            sample[index] += f"({match['unit']})"

        return samples_to_table(labels, samples, delimiter)
