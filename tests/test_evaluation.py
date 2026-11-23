"""Synthetic-only tests. CSV and PNG outputs are created in temporary directories."""

import csv
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.evaluate import (
    ACTIVITY_NAMES,
    analyze_misclassifications,
    compute_confusion_matrix,
    evaluate_model,
    plot_confusion_matrix,
    plot_loss_curve,
)
from src.experiment_results import RESULT_FIELDS, append_experiment_result


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = list(range(6))
        result = evaluate_model(labels, labels, label_base=0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(list(result["per_class"]), labels)
        for label, name in enumerate(ACTIVITY_NAMES):
            self.assertEqual(
                result["per_class"][label],
                {"name": name, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 1},
            )
        for average in ("macro_avg", "weighted_avg"):
            self.assertEqual(
                result[average], {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 6}
            )

    def test_partial_predictions_and_missing_class(self):
        result = evaluate_model([1, 1, 1, 2], [1, 1, 2, 1], label_base=1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["f1_score"], 2 / 3)
        self.assertAlmostEqual(result["macro_avg"]["f1_score"], 1 / 9)
        self.assertAlmostEqual(result["weighted_avg"]["f1_score"], 0.5)

        no_predictions = evaluate_model([1, 2], [1, 1], label_base=1)
        self.assertEqual(
            no_predictions["per_class"][2],
            {
                "name": "WALKING_UPSTAIRS", "precision": 0.0,
                "recall": 0.0, "f1_score": 0.0, "support": 1,
            },
        )
        self.assertEqual(no_predictions["per_class"][6]["support"], 0)

    def test_explicit_label_base_and_optional_names(self):
        index_metrics = evaluate_model([1], [1], label_base=0)
        raw_metrics = evaluate_model([1], [1], label_base=1)
        self.assertEqual(index_metrics["per_class"][1]["name"], "WALKING_UPSTAIRS")
        self.assertEqual(raw_metrics["per_class"][1]["name"], "WALKING")
        custom_names = tuple(f"Activity {i}" for i in range(6))
        result = evaluate_model([6], [6], label_base=1, class_names=custom_names)
        self.assertEqual(result["per_class"][6]["name"], custom_names[-1])

    def test_invalid_labels_and_names(self):
        for y_true, y_pred, base, error in (
            ([0, 1], [0], 0, "same length"),
            ([], [], 0, "nonempty"),
            ([[0]], [0], 0, "one-dimensional"),
            ([0], [6], 0, "integer class IDs"),
            ([0], [1], 1, "integer class IDs"),
            ([1.5], [1], 1, "integer class IDs"),
            (["1"], [1], 1, "integer class IDs"),
            ([True], [1], 1, "integer class IDs"),
            ([1], [1], True, "label_base"),
            ([1], [1], 2, "label_base"),
        ):
            with self.subTest(y_true=y_true, y_pred=y_pred, base=base):
                with self.assertRaisesRegex(ValueError, error):
                    evaluate_model(y_true, y_pred, label_base=base)
        with self.assertRaisesRegex(ValueError, "class_names"):
            evaluate_model([1], [1], label_base=1, class_names=["duplicate"] * 6)


class ConfusionMatrixTests(unittest.TestCase):
    def test_shape_order_counts_and_both_encodings(self):
        expected = np.zeros((6, 6), dtype=int)
        expected[0, 5] = 1
        expected[3, 3] = 1
        expected[3, 4] = 2
        expected[4, 3] = 1
        for base in (0, 1):
            with self.subTest(label_base=base):
                y_true = np.array([0, 3, 3, 3, 4]) + base
                y_pred = np.array([5, 3, 4, 4, 3]) + base
                matrix = compute_confusion_matrix(y_true, y_pred, label_base=base)
                self.assertEqual(matrix.shape, (6, 6))
                np.testing.assert_array_equal(matrix, expected)
                self.assertEqual(matrix.sum(), len(y_true))

    def test_plot_uses_activity_order_on_both_axes_and_saves_png(self):
        from matplotlib.figure import Figure

        figure = Figure()
        with TemporaryDirectory() as temporary:
            with patch("src.evaluate.Figure", return_value=figure):
                actual = plot_confusion_matrix(
                    [4, 5], [5, 4], label_base=1, output_path=Path(temporary) / "plots" / "matrix"
                )
            self.assertEqual(actual.suffix, ".png")
            self.assertEqual(actual.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            axes = figure.axes[0]
            self.assertEqual(tuple(tick.get_text() for tick in axes.get_xticklabels()), ACTIVITY_NAMES)
            self.assertEqual(tuple(tick.get_text() for tick in axes.get_yticklabels()), ACTIVITY_NAMES)
            self.assertEqual(axes.get_xlabel(), "Predicted label")
            self.assertEqual(axes.get_ylabel(), "True label")


class LossCurveTests(unittest.TestCase):
    def test_plot_starts_at_epoch_one_and_saves_png(self):
        from matplotlib.figure import Figure

        figure = Figure()
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "plots" / "synthetic_loss.png"
            with patch("src.evaluate.Figure", return_value=figure):
                path = plot_loss_curve(
                    [1.1, 0.6, 0.3], [1.2, 0.8, 0.5],
                    experiment_name="Synthetic", output_path=destination,
                )
            self.assertEqual(path, destination)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            axes = figure.axes[0]
            for line, loss, name in zip(
                axes.lines, ([1.1, 0.6, 0.3], [1.2, 0.8, 0.5]),
                ("Training Loss", "Validation Loss"),
            ):
                np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
                np.testing.assert_array_equal(line.get_ydata(), loss)
                self.assertEqual(line.get_label(), name)
            self.assertEqual(axes.get_xlabel(), "Epoch")
            self.assertEqual(axes.get_ylabel(), "Loss")
            self.assertIn("Synthetic", axes.get_title())
            self.assertIsNotNone(axes.get_legend())

    def test_invalid_histories(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.png"
            for train, val, error in (
                ([1, 2], [1], "same number of epochs"),
                ([], [], "nonempty"),
                ([1], [np.nan], "finite numbers"),
                ([np.inf], [1], "finite numbers"),
                (["1"], [1], "finite numbers"),
                ([[1]], [1], "one-dimensional"),
            ):
                with self.subTest(train=train, val=val):
                    with self.assertRaisesRegex(ValueError, error):
                        plot_loss_curve(train, val, output_path=path)
            self.assertFalse(path.exists())


class MisclassificationTests(unittest.TestCase):
    def test_no_errors(self):
        result = analyze_misclassifications([0, 1], [0, 1], label_base=0)
        self.assertEqual(result["total_errors"], 0)
        self.assertEqual(result["error_rate"], 0.0)
        self.assertEqual(result["confusion_pairs"], [])
        self.assertEqual(result["per_class"][0]["error_rate"], 0.0)
        self.assertIsNone(result["per_class"][2]["error_rate"])

    def test_directed_pairs_and_imbalanced_class_rates(self):
        result = analyze_misclassifications(
            [4, 4, 4, 4, 5, 6, 6], [5, 5, 4, 4, 4, 6, 1], label_base=1
        )
        self.assertEqual(result["total_samples"], 7)
        self.assertEqual(result["total_errors"], 4)
        self.assertAlmostEqual(result["error_rate"], 4 / 7)
        self.assertEqual(
            result["per_class"][4],
            {"name": "SITTING", "support": 4, "error_count": 2, "error_rate": 0.5},
        )
        self.assertEqual(result["per_class"][5]["error_rate"], 1.0)
        self.assertEqual(
            result["confusion_pairs"][0],
            {
                "true_label": 4, "true_name": "SITTING", "predicted_label": 5,
                "predicted_name": "STANDING", "count": 2,
            },
        )
        self.assertEqual(
            [(pair["true_label"], pair["predicted_label"]) for pair in result["confusion_pairs"]],
            [(4, 5), (5, 4), (6, 1)],
        )


class ExperimentRecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "synthetic_experiment_results.csv"
        self.record = {
            "experiment_id": "synthetic-1", "model": "Synthetic", "experiment_type": "Synthetic check"
        }

    def read_rows(self):
        with self.path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def test_creates_and_appends_separate_validation_and_test_rows(self):
        validation = {
            **self.record,
            "evaluation_split": "validation",
            "metric_average": "macro",
            "hyperparameters": {"learning_rate": 0.001},
            "random_seed": 42,
            "train_loss": 0.3,
            "val_loss": 0.5,
            "accuracy": 0.75,
            "f1_score": 0.5,
        }
        self.assertEqual(append_experiment_result(validation, output_path=self.path), self.path)
        append_experiment_result(
            {**self.record, "evaluation_split": "test", "accuracy": 0.0}, output_path=self.path
        )
        first, second = self.read_rows()
        self.assertEqual(tuple(first), RESULT_FIELDS)
        self.assertEqual(first["accuracy"], "0.75")
        self.assertEqual(first["train_loss"], "0.3")
        self.assertEqual(first["val_loss"], "0.5")
        self.assertEqual(first["random_seed"], "42")
        self.assertEqual(json.loads(first["hyperparameters"]), {"learning_rate": 0.001})
        self.assertEqual(first["metric_average"], "macro")
        self.assertEqual(first["evaluation_split"], "validation")
        self.assertEqual(second["evaluation_split"], "test")
        self.assertEqual(second["accuracy"], "0.0")
        self.assertEqual(second["precision"], "")
        self.assertEqual(second["val_loss"], "")
        self.assertEqual(second["notes"], "")

    def test_upgrades_legacy_csv_and_keeps_existing_values(self):
        header = (
            "experiment_id,date,model,dataset,preprocessing,random_seed,train_loss,val_loss,"
            "accuracy,precision,recall,f1_score,notes"
        )
        self.path.write_text(
            header + '\nsynthetic-old,,Synthetic,,,42,,,0.9,,,,"note, with\nline break"\n',
            encoding="utf-8",
        )
        old_row, = self.read_rows()
        append_experiment_result(self.record, output_path=self.path)
        first, second = self.read_rows()
        self.assertEqual({field: first[field] for field in old_row}, old_row)
        self.assertEqual(first["evaluation_split"], "")
        self.assertEqual(first["metric_average"], "")
        self.assertEqual(second["experiment_id"], "synthetic-1")
        self.assertEqual(second["accuracy"], "")

    def test_invalid_metrics_leave_existing_file_untouched(self):
        append_experiment_result(self.record, output_path=self.path)
        before = self.path.read_bytes()
        for supplied, error in (
            ({"accuracy": 0.5}, "evaluation_split is required"),
            ({"evaluation_split": "training"}, "evaluation_split must"),
            ({"evaluation_split": "test", "f1_score": 0.5}, "metric_average is required"),
            ({"evaluation_split": "test", "accuracy": 75}, "between 0 and 1"),
            ({"train_loss": np.nan}, "finite number"),
            ({"random_seed": "42"}, "must be an integer"),
            ({"recoll": 1}, "Unknown experiment result fields"),
        ):
            with self.subTest(supplied=supplied):
                with self.assertRaisesRegex(ValueError, error):
                    append_experiment_result({**self.record, **supplied}, output_path=self.path)
                self.assertEqual(self.path.read_bytes(), before)

    def test_malformed_existing_csv_is_preserved(self):
        content = ",".join(RESULT_FIELDS).encode() + b"\nsynthetic-incomplete,only-two-columns\n"
        self.path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "Malformed experiment results row"):
            append_experiment_result(self.record, output_path=self.path)
        self.assertEqual(self.path.read_bytes(), content)

    def test_failed_replacement_preserves_the_previous_csv(self):
        append_experiment_result(self.record, output_path=self.path)
        before = self.path.read_bytes()
        existing_files = set(self.path.parent.iterdir())
        with patch("src.experiment_results.os.replace", side_effect=OSError("Synthetic failure")):
            with self.assertRaisesRegex(OSError, "Synthetic failure"):
                append_experiment_result(self.record, output_path=self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.path.parent.iterdir()), existing_files)

    def test_concurrent_appends_preserve_all_rows(self):
        def append(experiment_id):
            append_experiment_result(
                {**self.record, "experiment_id": experiment_id}, output_path=self.path
            )

        experiment_ids = [f"synthetic-{i}" for i in range(12)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(append, experiment_ids))
        self.assertCountEqual([row["experiment_id"] for row in self.read_rows()], experiment_ids)


if __name__ == "__main__":
    unittest.main()
