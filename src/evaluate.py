"""Framework-independent evaluation of six-class UCI HAR predictions."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ACTIVITY_NAMES = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
FIGURES_DIR = Path(__file__).resolve().parents[1] / "experiments" / "figures"


def _class_ids(label_base):
    if (
        isinstance(label_base, (bool, np.bool_))
        or not isinstance(label_base, (int, np.integer))
        or label_base not in (0, 1)
    ):
        raise ValueError("label_base must be 0 (model IDs) or 1 (raw UCI IDs).")
    return list(range(int(label_base), int(label_base) + len(ACTIVITY_NAMES)))


def _class_names(class_names):
    if class_names is None:
        return ACTIVITY_NAMES
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be a sequence of six activity names.")
    names = tuple(class_names)
    if (
        len(names) != len(ACTIVITY_NAMES)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("class_names must contain six distinct, nonempty names in ID order.")
    return names


def _label_arrays(y_true, y_pred, class_ids):
    arrays = []
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        labels = np.asarray(values)
        if labels.ndim != 1 or not labels.size:
            raise ValueError(f"{name} must be a nonempty one-dimensional array.")
        if labels.dtype.kind not in "iu" or not np.isin(labels, class_ids).all():
            raise ValueError(
                f"{name} must contain integer class IDs from {class_ids[0]} to {class_ids[-1]}."
            )
        arrays.append(labels.astype(np.int64, copy=False))
    if arrays[0].size != arrays[1].size:
        raise ValueError("y_true and y_pred must have the same length.")
    return arrays


def _metric_values(row):
    return {
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1-score"],
        "support": int(row["support"]),
    }


def evaluate_model(y_true, y_pred, *, label_base, class_names=None):
    """Return accuracy, per-class metrics keyed by ID, macro and weighted averages."""
    class_ids = _class_ids(label_base)
    names = _class_names(class_names)
    y_true, y_pred = _label_arrays(y_true, y_pred, class_ids)
    report = classification_report(
        y_true, y_pred, labels=class_ids, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {"name": name, **_metric_values(report[str(label)])}
            for label, name in zip(class_ids, names)
        },
        "macro_avg": _metric_values(report["macro avg"]),
        "weighted_avg": _metric_values(report["weighted avg"]),
    }


def compute_confusion_matrix(y_true, y_pred, *, label_base):
    """Return a 6×6 count matrix with true rows and predicted columns in ID order."""
    class_ids = _class_ids(label_base)
    y_true, y_pred = _label_arrays(y_true, y_pred, class_ids)
    return confusion_matrix(y_true, y_pred, labels=class_ids)


def _save_figure(figure, output_path):
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    label_base,
    class_names=None,
    output_path=FIGURES_DIR / "confusion_matrix.png",
):
    """Save a labeled confusion matrix and return its path."""
    names = _class_names(class_names)
    matrix = compute_confusion_matrix(y_true, y_pred, label_base=label_base)
    figure = Figure(figsize=(9, 7), constrained_layout=True)
    axes = figure.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(
        ax=axes, cmap="Blues", values_format="d", xticks_rotation=45
    )
    axes.set_title("Confusion Matrix")
    return _save_figure(figure, output_path)


def _loss_array(values, name):
    loss = np.asarray(values)
    if (
        loss.ndim != 1
        or not loss.size
        or loss.dtype.kind not in "iuf"
        or not np.isfinite(loss).all()
    ):
        raise ValueError(f"{name} must be a nonempty one-dimensional array of finite numbers.")
    return loss


def plot_loss_curve(
    train_loss,
    val_loss,
    *,
    experiment_name=None,
    output_path=FIGURES_DIR / "loss_curve.png",
):
    """Save paired training and validation losses, with one value per epoch."""
    train_loss = _loss_array(train_loss, "train_loss")
    val_loss = _loss_array(val_loss, "val_loss")
    if train_loss.size != val_loss.size:
        raise ValueError("train_loss and val_loss must contain the same number of epochs.")

    figure = Figure(figsize=(8, 5), constrained_layout=True)
    axes = figure.subplots()
    epochs = np.arange(1, train_loss.size + 1)
    axes.plot(epochs, train_loss, label="Training Loss")
    axes.plot(epochs, val_loss, label="Validation Loss")
    title = f"Loss Curve — {experiment_name}" if experiment_name else "Loss Curve"
    axes.set(title=title, xlabel="Epoch", ylabel="Loss")
    axes.xaxis.get_major_locator().set_params(integer=True)
    axes.legend()
    return _save_figure(figure, output_path)


def analyze_misclassifications(y_true, y_pred, *, label_base, class_names=None):
    """Return error totals, rates by true class, and ranked directed confusion pairs."""
    names = _class_names(class_names)
    matrix = compute_confusion_matrix(y_true, y_pred, label_base=label_base)
    class_ids = range(int(label_base), int(label_base) + len(ACTIVITY_NAMES))
    per_class = {}
    pairs = []
    for row, label in enumerate(class_ids):
        support = int(matrix[row].sum())
        error_count = support - int(matrix[row, row])
        per_class[label] = {
            "name": names[row],
            "support": support,
            "error_count": error_count,
            "error_rate": error_count / support if support else None,
        }
        for column, predicted_label in enumerate(class_ids):
            count = int(matrix[row, column])
            if row != column and count:
                pairs.append({
                    "true_label": label,
                    "true_name": names[row],
                    "predicted_label": predicted_label,
                    "predicted_name": names[column],
                    "count": count,
                })

    pairs.sort(key=lambda pair: (-pair["count"], pair["true_label"], pair["predicted_label"]))
    total_samples = int(matrix.sum())
    total_errors = total_samples - int(matrix.trace())
    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": total_errors / total_samples,
        "per_class": per_class,
        "confusion_pairs": pairs,
    }
