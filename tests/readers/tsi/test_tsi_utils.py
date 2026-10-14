import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pypana.readers.exceptions.read_error import ReadError
from pypana.readers.tsi.utils import is_basic_tsi_format_file
from pypana.readers.utils import DELIMITERS

ANCHORS = ["Classifier Model\t3082", "Units", "Sample #"]


def _write_file(directory: str, delimiter: str, samples: list[int]) -> Path:
    lines = [
        f"Classifier Model{delimiter}3082",
        f"Units{delimiter}dw/dlogDp",
        delimiter.join(["Sample #", "Date", "Start Time"]),
        *[delimiter.join([str(s), "01/01/2024", "10:00:00"]) for s in samples],
    ]
    path = Path(directory) / "file.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    return path


@given(delimiter=st.sampled_from(DELIMITERS), n=st.integers(1, 20))
def test_basic_format_with_any_delimiter(delimiter: str, n: int) -> None:
    """The anchors match with every supported delimiter."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, delimiter, list(range(1, n + 1)))

        assert is_basic_tsi_format_file(path, ANCHORS)


@given(delimiter=st.sampled_from(DELIMITERS))
def test_samples_out_of_order_raise(delimiter: str) -> None:
    """Scans that are not in order raise."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, delimiter, [2, 1])

        with pytest.raises(ReadError):
            is_basic_tsi_format_file(path, ANCHORS)


def test_other_classifier_model_is_not_read() -> None:
    """A different classifier model does not match."""
    with tempfile.TemporaryDirectory() as directory:
        path = _write_file(directory, "\t", [1, 2])

        assert not is_basic_tsi_format_file(
            path, ["Classifier Model\t3080", "Sample #"]
        )
