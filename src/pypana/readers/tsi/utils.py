"""Utility functions for reading specific to TSI."""

import io
import itertools
from pathlib import Path

import pandas as pd

from pypana.readers.exceptions.read_error import ReaderNotImplementedError, ReadError
from pypana.readers.utils import (
    check_anchors_in_file,
    check_samples_in_integer_order_in_file,
    detect_delimiter,
)


def read_column_lines(
    path: Path,
    header_pos: int,
    delimiter: str,
    *,
    single_cell_labels: tuple[str, ...] = (),
    encoding: str = "utf-8",
) -> tuple[list[str], list[list[str]]]:
    """Reads an export with one column per sample as text, one list of values per sample.

    Args:
        path: The path to the input file.
        header_pos: The line index of the line holding the sample numbers.
        delimiter: The delimiter separating the columns of the file.
        single_cell_labels: Labels of lines that hold no values and become empty columns.
        encoding: Encoding of the file.

    Returns:
        The labels and the values of each sample in the order of the labels.

    Raises:
        ReaderNotImplementedError: If there is more than one column per sample.
        ReadError: If the sample numbers or the number of values per line are inconsistent.
    """
    with Path.open(path, "r", encoding=encoding) as f:
        lines = [line.rstrip("\r\n") for line in itertools.islice(f, header_pos, None)]

    rows = [line.split(delimiter) for line in lines if line.strip()]
    sample_ids = rows[0][1:]

    if "" in sample_ids:
        raise ReaderNotImplementedError(
            "Column exports with more than one column per sample (RawData) are not supported."
        )

    ids = [int(i) for i in sample_ids if i.isdigit()]

    if len(ids) != len(sample_ids) or any(a >= b for a, b in itertools.pairwise(ids)):
        raise ReadError(
            message="The sample numbers are not strictly increasing integers.",
            path=path,
        )

    labels: list[str] = []
    values: list[list[str]] = []

    for row in rows:
        if row[0] in single_cell_labels and len(row) == 1:
            values.append([""] * len(ids))
        elif len(row) == len(ids) + 1:
            values.append(row[1:])
        else:
            raise ReadError(
                message=f"Row '{row[0]}' has {len(row) - 1} values for {len(ids)} samples.",
                path=path,
            )

        labels.append(row[0])

    return labels, [list(sample) for sample in zip(*values, strict=True)]


def samples_to_table(
    labels: list[str], samples: list[list[str]], delimiter: str
) -> pd.DataFrame:
    """Parses the text values of the samples with the same pandas call as a row oriented export.

    Args:
        labels: The column labels.
        samples: The values of each sample in the order of the labels.
        delimiter: The delimiter separating the columns of the file.

    Returns:
        The table with one row per sample.
    """
    text = "\n".join(delimiter.join(line) for line in [labels, *samples])

    # best way found to pass string to pandas
    return pd.read_table(io.StringIO(text), sep=delimiter, skip_blank_lines=True)


def read_metadata_block(
    path: Path,
    header_start: str,
    delimiter: str,
    *,
    encoding: str = "utf-8",
) -> tuple[int, dict[str, str]]:
    """Scans the metadata block above the table header.

    Args:
        path: The path to the input file.
        header_start: The start of the table header line, with tabs as delimiters.
        delimiter: The delimiter separating the columns of the file.
        encoding: Encoding of the file.

    Returns:
        The header line index and the metadata as a mapping from label to first value.

    Raises:
        ReadError: If the table header cannot be found.
    """
    header_start = header_start.replace("\t", delimiter)
    header_pos: int | None = None
    metadata: dict[str, str] = {}

    with Path.open(path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if line.startswith(header_start):
                header_pos = i
                break

            fields = line.rstrip("\r\n").split(delimiter)
            metadata[fields[0]] = fields[1].strip() if len(fields) > 1 else ""

    if header_pos is None:
        raise ReadError(
            message="The file does not contain the start of the table header.",
            path=path,
        )

    return header_pos, metadata


def is_basic_tsi_format_file(
    path: Path,
    anchors: list[str],
    *,
    check_sample_order: bool = True,
    encoding: str = "utf-8",
) -> bool:
    """Check if a TSI file is basic format.

    Returns:
        bool: True if basic format, False otherwise.

    Raises:
        ReadError: if likely TSI format but something is wrong.
    """
    if not path.is_file():
        return False

    delimiter = detect_delimiter(path, encoding=encoding)

    if delimiter is None:
        return False

    all_anchors_found, last_line = check_anchors_in_file(
        path,
        [anchor.replace("\t", delimiter) for anchor in anchors],
        encoding=encoding,
    )

    if not all_anchors_found:
        return False

    if not check_sample_order:
        return True

    if not check_samples_in_integer_order_in_file(
        path,
        last_line + 1,  # start in line after header
        delimiter=delimiter,
        encoding=encoding,
    ):
        raise ReadError(
            message="The scans appear to not be in order",
            path=path,
        )

    return True
