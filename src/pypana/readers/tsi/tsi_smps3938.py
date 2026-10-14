"""Implementation of a Reader for the TSI Universal-Scanning Mobility Particle Sizer (SMPS) 3938.

This module provides the corresponding reader for the produced files of a TSI SMPS 3938.

References:
    https://tsi.com/products/particle-sizers/scanning-mobility-particle-sizer-spectrometers/general-scanning-mobility-particle-sizer-smps-3938
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import Index

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


class TSISMPS3938InstrumentReader(BaseInstrumentReader):
    """Instrument reader for TSI SMPS 3938."""

    _encoding = "iso-8859-1"
    _device_name = "TSI SMPS 3938"
    _input_type = InputType.FILE

    _REPORTED_UNITS = "dw/dlogDp"
    _REPORTED_WEIGHTS = {
        "Number": (Quantity.NUMBER, "#/cm³"),
        "Surface": (Quantity.SURFACE, "nm²/cm³"),
        "Volume": (Quantity.VOLUME, "nm³/cm³"),
    }
    _TOTAL_CONC_PATTERN = re.compile(r"^Total Conc\. \((.+)\)$")
    _SIZE_SCALING_FACTOR = UnitScale.NANO.value
    _HEADER_START = "Sample #\tDate\tStart Time"
    _SAMPLES_AS_ROWS = True
    _DIAMETER_LABEL = "Diameter Midpoint (nm)"
    _FIRST_TRAILING_COLUMN = "Scan Time (s)"
    _DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"
    _CHANNELS_PER_DECADE_KEY = "Channels/Decade"
    _FLAGS = {
        "Multiple Charge Correction": "multiple_charge_correction",
        "Nanoparticle Agglomerate Mobility Analysis": "nanoparticle_agglomerate_mobility_analysis",
        "Diffusion Correction": "diffusion_correction",
    }
    _BOOLEANS = {"TRUE": True, "FALSE": False}
    _MIDPOINT_RESOLUTION = 0.1

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Checks whether a given path may include a TSI SMPS 3938 output file that can be read.

        Current checks include:
            - whether the path is a file
            - the 3082 classifier, channel resolution, Units and Weight are given
            - the scan lines start with continuous integers

        Args:
            path: The path to the input file.

        Returns:
            Whether the read test succeeded when applying the TSI SMPS 3938 format.

        Raises:
            ReadError: If confident enough that the input is from TSI SMPS 3938, but the data suggests otherwise.
                This may happen because the input files were manually edited in unsafe places or this package
                does not yet fully implement this device's formats.
                Note: the absence of ReadError in this method does not guarantee the input is parseable.
        """
        anchors = [
            "Classifier Model\t3082",
            "Channels/Decade",
            "Units",
            "Weight",
            cls._HEADER_START,
        ]

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
            ReaderNotImplementedError: If the file does not report Number-weighted dN/dlogDp data.
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

        units = metadata.get("Units")
        weight = metadata.get("Weight")

        if units != self._REPORTED_UNITS or weight not in self._REPORTED_WEIGHTS:
            raise ReaderNotImplementedError(
                f"Only {self._REPORTED_UNITS} data weighted by {', '.join(self._REPORTED_WEIGHTS)} "
                f"is implemented for TSI SMPS 3938 (got weight={weight!r}, units={units!r})."
            )

        quantity, expected_unit = self._REPORTED_WEIGHTS[weight]
        scale = self._SIZE_SCALING_FACTOR**quantity.moment
        channels_per_decade = self._read_channels_per_decade(metadata)
        flags = self._read_flags(metadata)

        try:
            data = self._read_table(delimiter, header_pos)

            diameter_loc = data.columns.get_loc(self._DIAMETER_LABEL)
            assert isinstance(diameter_loc, int)
            trailing_loc = data.columns.get_loc(self._FIRST_TRAILING_COLUMN)
            assert isinstance(trailing_loc, int)

            bin_columns = data.columns[diameter_loc + 1 : trailing_loc]
            bin_boundaries = self._build_bin_boundaries(
                bin_columns, channels_per_decade
            )
            axis = BinAxis(
                bin_boundaries=bin_boundaries.copy(), diameter_type="mobility"
            )

            trailing_columns = data.columns[trailing_loc:].to_list()

        except (FileNotFoundError, ValueError, KeyError) as e:
            raise ReadError(f"{e}") from e

        reported_unit = self._reported_unit(trailing_columns)

        if reported_unit != expected_unit:
            raise ReadError(
                message=f"The total concentration is reported in {reported_unit!r}, expected {expected_unit!r}.",
                path=self._path,
            )

        measurements: dict[int, Measurement] = {}

        for row in data.to_dict("records"):
            try:
                scan_nr = int(row["Sample #"])
                time = datetime.strptime(
                    f"{row['Date']} {row['Start Time']}", self._DATETIME_FORMAT
                )
                delta_dlogdp = (
                    np.array([row[col] for col in bin_columns], dtype=float) * scale
                )

                distribution = SizeDistribution(
                    quantity=quantity,
                    axis=axis,
                    delta_dlogdp=delta_dlogdp,
                )

                other_info = other_columns_to_dict(row, trailing_columns)

                measurements[scan_nr - 1] = Measurement(  # 0-based indexing
                    scan_nr=scan_nr,
                    time=time,
                    axis=axis,
                    distributions={quantity: distribution},
                    other=other_info,
                )
            except (ValueError, AttributeError, KeyError) as e:
                raise ReadError(f"{e}") from e

        return InstrumentData(
            measurements=measurements,
            device_name=self._device_name,
            file_path=self._path,
            other_info={
                "channels_per_decade": channels_per_decade,
                "units": units,
                "weight": weight,
                **flags,
            },
        )

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

    @classmethod
    def _reported_unit(cls, columns: list[str]) -> str | None:
        """Extracts the unit.

        Args:
            columns: The trailing column headers.

        Returns:
            The reported unit or None.
        """
        for column in columns:
            match = cls._TOTAL_CONC_PATTERN.match(column)

            if match:
                return match.group(1)

        return None

    def _read_channels_per_decade(self, metadata: dict[str, str]) -> int:
        """Reads the channel resolution from the metadata.

        Args:
            metadata: The metadata block of the file.

        Returns:
            The number of channels per decade.

        Raises:
            ReadError: If the file does not report a valid channel resolution.
        """
        try:
            return int(metadata[self._CHANNELS_PER_DECADE_KEY])
        except (KeyError, ValueError) as e:
            raise ReadError(
                message="The file does not report a valid 'Channels/Decade'.",
                path=self._path,
            ) from e

    def _read_flags(self, metadata: dict[str, str]) -> dict[str, bool]:
        """Reads the correction flags from the metadata.

        Args:
            metadata: The metadata block of the file.

        Returns:
            The correction flags by name.

        Raises:
            ReadError: If a flag is missing or neither TRUE nor FALSE.
        """
        flags: dict[str, bool] = {}

        for label, name in self._FLAGS.items():
            value = metadata.get(label)

            if value not in self._BOOLEANS:
                raise ReadError(
                    message=f"Expected TRUE or FALSE for '{label}', got {value!r}.",
                    path=self._path,
                )

            flags[name] = self._BOOLEANS[value]

        return flags

    @classmethod
    def _build_bin_boundaries(
        cls, bin_columns: Index, channels_per_decade: int
    ) -> FloatArray:
        """Builds the decade aligned boundary grid in meters.

        Args:
            bin_columns: The concentration column headers.
            channels_per_decade: The instrument's channel resolution.

        Returns:
            The ``n + 1`` bin boundaries in meters.

        Raises:
            ValueError: If the header midpoints do not lie on the decade aligned grid.
        """
        midpoints_file = np.array(bin_columns, dtype=float)
        channels = np.arange(len(midpoints_file))

        offset = np.mean(np.log10(midpoints_file) - channels / channels_per_decade)
        first = round(float(offset) * channels_per_decade - 0.5)

        midpoints = 10.0 ** ((first + 0.5 + channels) / channels_per_decade)
        tolerance = cls._MIDPOINT_RESOLUTION / 2 * (1 + 1e-6)

        if np.any(np.abs(midpoints - midpoints_file) > tolerance):
            # if this fails, this entire calculation is wrong!
            raise ValueError("The bin midpoints do not lie on a decade aligned grid.")

        edges = 10.0 ** (
            (first + np.arange(len(midpoints_file) + 1)) / channels_per_decade
        )

        return edges * cls._SIZE_SCALING_FACTOR


class TSISMPS3938ColumnInstrumentReader(TSISMPS3938InstrumentReader):
    """Instrument reader for TSI SMPS 3938 files with one column per sample."""

    _HEADER_START = "Sample #\t1"
    _SAMPLES_AS_ROWS = False
    _SEPARATOR_LABEL = "Diameter Midpoint (nm)"
    _SUPPORTED_DELIMITER = "\t"

    def _read_table(self, delimiter: str, header_pos: int) -> pd.DataFrame:
        """Reads the data table and transposes it to one row per sample.

        Args:
            delimiter: The delimiter separating the columns of the file.
            header_pos: The line index of the table header.

        Returns:
            The table with one row per sample, as in the row oriented export.

        Raises:
            ReaderNotImplementedError: If the delimiter is not a tab or there is more than one column per sample.
            ReadError: If the sample numbers or the number of values per row are inconsistent.
        """
        if delimiter != self._SUPPORTED_DELIMITER:
            raise ReaderNotImplementedError(
                "Only tab delimited column exports are implemented for TSI SMPS 3938."
            )

        labels, samples = read_column_lines(
            self._path,
            header_pos,
            delimiter,
            single_cell_labels=(self._SEPARATOR_LABEL,),
            encoding=self._encoding,
        )

        return samples_to_table(labels, samples, delimiter)
