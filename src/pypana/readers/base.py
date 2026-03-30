"""Base definition for instrument data readers.

This module provides the abstract base class for all instrument-specific readers.
It includes a registry of all subclasses to enable automatic format discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

type InstrumentReaderList = list[type[BaseInstrumentReader]]


class BaseInstrumentReader(ABC):
    """Base instrument reader for all devices.

    Attributes:
        _subclass_registry (InstrumentReaderList): Internal registry of all available instrument reader classes.
    """

    _is_router = False
    _subclass_registry: InstrumentReaderList = []

    def __init__(self, path: Path) -> None:
        """Default constructor for all readers.

        Args:
            path: The path to the input file.
        """
        self.path = path

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__()
        if not cls._is_router and cls not in BaseInstrumentReader._subclass_registry:
            BaseInstrumentReader._subclass_registry.append(cls)

    @classmethod
    def _deregister(cls, target: type[BaseInstrumentReader]) -> None:
        """Remove a reader from the subclass registry. Internal use for testing."""
        if target in cls._subclass_registry:
            cls._subclass_registry.remove(target)

    @classmethod
    def registered_readers(cls) -> InstrumentReaderList:
        """Returns a copy of all registered readers.

        Returns:
            InstrumentReaderList: A list of classes that can be used for file type discovery.
        """
        return cls._subclass_registry.copy()

    @abstractmethod
    def can_read(self, path: Path | None) -> bool:
        """Check if this reader can read a given file. It indicates that this class is the correct reader.

        Args:
            path: The Path to the input file.

        Returns:
            Whether the read test succeeded.
        """
