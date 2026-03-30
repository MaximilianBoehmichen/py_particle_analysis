"""Definition of the intelligent SmartReader.

This module provides a class for automatically choosing the correct reader and therefore provides a real
instrument-agnostic interface to the user.
"""

from pathlib import Path
from typing import cast

from pypana.readers.base import BaseInstrumentReader, InstrumentReaderList
from pypana.readers.exceptions.read_error import ReadError
from pypana.readers.exceptions.too_many_options import TooManyOptionsError


class SmartReader(BaseInstrumentReader):
    """Smart Reader for automatically detecting and choosing the correct reader for a given input file."""

    _is_router = True

    def __new__(cls, path: Path, *args: object, **kwargs: object) -> "SmartReader":
        """Routes to the correct reader for a given input file and returns an instance of this class.

        This uses some unintuitive black magic that defies the principles of OOP.
        However, this is the only way to provide seamless initialization without further calls to a factory before
        the pipeline can continue like when the correct reader is initialized in the first place.

        Args:
            path: The path to the input file.
            *args: args
            **kwargs: kwargs
        """
        possible_readers: InstrumentReaderList = []

        for reader_class in BaseInstrumentReader.registered_readers():
            instance = super().__new__(reader_class)
            instance.__init__(path)  # type: ignore[misc]

            result = instance.can_read(path)

            if result:
                possible_readers.append(reader_class)

        if len(possible_readers) == 0:
            raise ReadError(
                message="No implemented reader can read this input file.", path=path
            )

        if len(possible_readers) > 1:
            raise TooManyOptionsError(
                message="The SmartReader cannot determine the correct reader.",
                path=path,
                possible_readers=possible_readers,
            )

        instance = super().__new__(possible_readers.pop())
        instance.__init__(path)  # type: ignore[misc]

        return cast("SmartReader", instance)

    def can_read(self, path: Path | None) -> bool:
        """Check if this reader can read a given file. It cannot!

        Args:
            path: The Path to the input file.

        Returns:
            Whether the read test succeeded.

        Raises:
            ReadError: always
        """
        raise ReadError(
            message="This class can only detect the correct reader, but itself is not one."
        )
