"""Append experiment results while preserving existing CSV records."""

import csv
import json
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Integral, Real
from pathlib import Path
from shutil import copymode
from tempfile import NamedTemporaryFile

RESULTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_results.csv"
LEGACY_FIELDS = (
    "experiment_id", "date", "model", "dataset", "preprocessing", "random_seed",
    "train_loss", "val_loss", "accuracy", "precision", "recall", "f1_score", "notes",
)
RESULT_FIELDS = LEGACY_FIELDS + (
    "experiment_type", "hyperparameters", "evaluation_split", "metric_average",
)


@contextmanager
def _lock_csv(path):
    # The lock file stays in place so the lock survives replacement of the CSV.
    with path.with_name(f".{path.name}.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _read_existing_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        fields = next(reader, None)
        if fields is None:
            return []
        if tuple(fields) not in (LEGACY_FIELDS, RESULT_FIELDS):
            raise ValueError(f"Unrecognized experiment results header in {path}.")
        rows = []
        for values in reader:
            if not values:
                continue
            if len(values) != len(fields):
                raise ValueError(f"Malformed experiment results row at line {reader.line_num}.")
            rows.append(dict(zip(fields, values)))
        return rows


def _prepare_record(record):
    if not isinstance(record, Mapping) or any(not isinstance(field, str) for field in record):
        raise ValueError("record must be a mapping with string field names.")
    row = dict(record)
    unknown_fields = row.keys() - set(RESULT_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unknown experiment result fields: {', '.join(sorted(unknown_fields))}.")
    for field in ("experiment_id", "model", "experiment_type"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string.")

    split = row.get("evaluation_split")
    average = row.get("metric_average")
    averaged_metrics = ("precision", "recall", "f1_score")
    metric_fields = ("accuracy", *averaged_metrics)
    if split not in (None, "", "validation", "test"):
        raise ValueError("evaluation_split must be 'validation' or 'test'.")
    if average not in (None, "", "macro", "weighted"):
        raise ValueError("metric_average must be 'macro' or 'weighted'.")
    if any(row.get(metric) not in (None, "") for metric in metric_fields) and not split:
        raise ValueError("evaluation_split is required when recording metrics.")
    if any(row.get(metric) not in (None, "") for metric in averaged_metrics) and not average:
        raise ValueError("metric_average is required when recording precision, recall or F1.")

    for field in ("train_loss", "val_loss", *metric_fields):
        value = row.get(field)
        if value in (None, ""):
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        if field in metric_fields and not 0 <= value <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")
    seed = row.get("random_seed")
    if seed not in (None, "") and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise ValueError("random_seed must be an integer.")

    config = row.get("hyperparameters")
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("hyperparameters must be a mapping.")
        row["hyperparameters"] = json.dumps(dict(config), sort_keys=True, allow_nan=False)
    return row


def append_experiment_result(record, *, output_path=RESULTS_PATH):
    """Append one experiment row and return the CSV path.

    Provide the CSV fields as a mapping. Use one row per evaluation split and
    specify the averaging method for precision, recall and F1. A legacy header
    is upgraded on append, with the new fields blank for existing rows.
    """
    row = _prepare_record(record)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock_csv(path):
        rows = _read_existing_rows(path)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(row)
            if path.exists():
                copymode(path, temporary_path)
            # Readers see a complete CSV even if writing a row fails.
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return path
