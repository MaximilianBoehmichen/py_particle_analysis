import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pypana.readers.exceptions.read_error import ReadError
from pypana.readers.utils import (
    DELIMITERS,
    check_samples_in_integer_order_in_file,
    detect_delimiter,
)


def _table_lines(delimiter: str, samples: list[int]) -> list[str]:
    header = delimiter.join(["Sample #", "Date", "Start Time"])
    rows = [delimiter.join([str(s), "01/01/2024", "10:00:00"]) for s in samples]

    return ["Sample File\tC:\\a,b;c", header, *rows]


def _write_file(directory: str, lines: list[str]) -> Path:
    path = Path(directory) / "file.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    return path


@given(delimiter=st.sampled_from(DELIMITERS), n=st.integers(1, 20))
def test_detect_delimiter(delimiter: str, n: int) -> None:
    """The delimiter follows 'Sample #', whatever the metadata above contains."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, _table_lines(delimiter, list(range(1, n + 1))))

        assert detect_delimiter(path) == delimiter


@given(character=st.sampled_from(["|", " ", "x"]))
def test_detect_delimiter_unknown(character: str) -> None:
    """An unknown character after 'Sample #' yields None."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, [f"Sample #{character}Date"])

        assert detect_delimiter(path) is None


def test_detect_delimiter_without_header() -> None:
    """A file without a table header yields None."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, ["Units\tdw/dlogDp"])

        assert detect_delimiter(path) is None


def test_detect_delimiter_missing_file() -> None:
    """A missing file yields None."""
    assert detect_delimiter(Path("does_not_exist.txt")) is None


@given(
    delimiter=st.sampled_from(DELIMITERS),
    samples=st.lists(st.integers(1, 50), min_size=1, max_size=20),
)
def test_samples_in_integer_order(delimiter: str, samples: list[int]) -> None:
    """The check is True exactly if the samples strictly increase."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, _table_lines(delimiter, samples))

        expected = all(a < b for a, b in zip(samples, samples[1:], strict=False))

        assert (
            check_samples_in_integer_order_in_file(path, 2, delimiter=delimiter)
            == expected
        )


@given(delimiter=st.sampled_from(DELIMITERS))
def test_samples_not_starting_with_integer_raise(delimiter: str) -> None:
    """A row without a leading sample number raises."""
    with tempfile.TemporaryDirectory() as directory:
        lines = [*_table_lines(delimiter, [1]), delimiter.join(["x", "y"])]
        path = _write_file(directory, lines)

        with pytest.raises(ReadError):
            check_samples_in_integer_order_in_file(path, 2, delimiter=delimiter)
