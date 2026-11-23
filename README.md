# Human Activity Recognition (HAR) — Deep Learning Final Project

This repository is the implementation workspace for the **Human Activity Recognition (HAR)** final project.

## Project Overview

The project aims to build a deep learning pipeline for Human Activity Recognition. The specific dataset, activity labels, model architecture, and evaluation protocol should be documented here once confirmed.

## Repository Structure

```text
HAR_Project/
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── best_model/
├── experiments/
│   ├── experiment_results.csv
│   └── figures/
├── app/
│   └── interface.py
├── report/
│   └── final_report.pdf
├── main.py
├── README.txt
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Ltrn4600/DL-Final-Project.git
cd DL-Final-Project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\\Scripts\\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Place the training and testing data in `dataset/train/` and `dataset/test/` respectively. Do not commit private, restricted, or large raw datasets unless the team has confirmed that redistribution is permitted. Add the dataset source, licence, label definitions, and expected file format here.

## Workflow

1. **Preprocessing** — inspect, clean, transform, and split/prepare the data as required.
2. **Model** — define the selected neural network architecture.
3. **Training** — train the model and save the best checkpoint under `models/best_model/`.
4. **Evaluation** — calculate the selected metrics and save experiment results.
5. **Prediction** — load a trained checkpoint and predict activity labels for new input.
6. **Interface** — provide a user-facing interface after the input format and UI framework are decided.

The Python files currently contain starter scaffolds. Complete the TODO sections after confirming the dataset format and model design.

## Run

The project entry point is:

```bash
python main.py
```

Individual modules can also be developed and tested independently. Update this section with exact commands once the pipeline is implemented.

## Experiment Tracking

Record each experiment in `experiments/experiment_results.csv`. Store plots in `experiments/figures/`. Include the model configuration, data preprocessing choices, random seed, and evaluation metrics so results can be reproduced.

### Evaluation utilities

Pass matching one-dimensional arrays of integer class IDs to `src.evaluate`.
Set `label_base=1` for the raw UCI HAR labels or `label_base=0` for the
CNN's class indices. Both arrays must use the same encoding.

| Activity | Raw UCI ID | CNN index |
| --- | ---: | ---: |
| WALKING | 1 | 0 |
| WALKING_UPSTAIRS | 2 | 1 |
| WALKING_DOWNSTAIRS | 3 | 2 |
| SITTING | 4 | 3 |
| STANDING | 5 | 4 |
| LAYING | 6 | 5 |

The raw IDs come from `activity_labels.txt` in the [UCI HAR dataset](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones).
The CNN indices follow the activity order in `models/model_code/model_training.py`.

| Function | Result |
| --- | --- |
| `evaluate_model(y_true, y_pred, *, label_base, class_names=None)` | Accuracy, per-class metrics keyed by ID, and `macro_avg` and `weighted_avg`. Macro averages include all six classes, with zero for undefined metrics. Weighted averages use the number of true samples per class. |
| `compute_confusion_matrix(y_true, y_pred, *, label_base)` | A 6×6 count array in ascending ID order, with true classes on rows and predicted classes on columns. |
| `plot_confusion_matrix(y_true, y_pred, *, label_base, class_names=None, output_path=...)` | The path to the labeled plot. Supply six names in ID order to customize its axes. |
| `plot_loss_curve(train_loss, val_loss, *, experiment_name=None, output_path=...)` | The path to a plot starting at epoch 1. Both histories must have one finite numeric value per epoch and cover the same number of epochs. |
| `analyze_misclassifications(y_true, y_pred, *, label_base, class_names=None)` | Total errors, overall error rate, per-class counts and rates, and directed confusion pairs sorted by count. A class with zero true samples has an error rate of `None`. |
| `append_experiment_result(record, *, output_path=...)` from `src.experiment_results` | The path to the updated CSV after appending one record. |

Plot paths default to `experiments/figures/confusion_matrix.png` and
`experiments/figures/loss_curve.png`. Supply a distinct path for each run or split to keep its plots.

### Connect a trained CNN or MLP

Use one prediction per sample, in the same order as the true labels for the
selected split. For the Keras CNN, convert the output scores with
`scores.argmax(axis=1)`. For a PyTorch MLP, collect the class indices with
`logits.argmax(dim=1).detach().cpu().numpy()` and concatenate batches in label
order. Use `label_base=0` when the model outputs follow the CNN class order.
When comparing against raw UCI labels, convert the predictions to the raw IDs
with `+ 1` and use `label_base=1`. Keras supplies the loss arrays as
`history.history["loss"]` and `history.history["val_loss"]`. With a PyTorch
loop, pass the training and validation loss lists collected once per epoch.

The following example assumes the caller already has the matching arrays and
metadata from a validation run.

```python
from src.evaluate import (
    analyze_misclassifications,
    evaluate_model,
    plot_confusion_matrix,
    plot_loss_curve,
)
from src.experiment_results import append_experiment_result

metrics = evaluate_model(y_true, y_pred, label_base=0)
errors = analyze_misclassifications(y_true, y_pred, label_base=0)
plot_confusion_matrix(
    y_true, y_pred, label_base=0,
    output_path=f"experiments/figures/{run_id}_validation_confusion.png",
)
plot_loss_curve(
    train_loss, val_loss, experiment_name=run_id,
    output_path=f"experiments/figures/{run_id}_loss.png",
)
macro = metrics["macro_avg"]
append_experiment_result({
    "experiment_id": run_id,
    "model": model_name,
    "experiment_type": experiment_type,
    "hyperparameters": run_config,
    "random_seed": seed,
    "train_loss": train_loss[-1],
    "val_loss": val_loss[-1],
    "evaluation_split": "validation",
    "metric_average": "macro",
    "accuracy": metrics["accuracy"],
    "precision": macro["precision"],
    "recall": macro["recall"],
    "f1_score": macro["f1_score"],
})
```

Record validation and test metrics on separate rows. Use the same
`experiment_id` for both rows of a single run, and include
`evaluation_split` whenever recording metrics. Specify `macro` or
`weighted` when recording precision, recall or F1. Metrics are fractions
from 0 to 1. Use validation results for hyperparameter comparisons and
record the held-out test metrics after choosing the model configuration.
The example records the last epoch's losses. When reporting losses from a
selected checkpoint, supply the losses from the intended evaluation or epoch
and identify that choice in `notes`.

The CSV keeps the original columns in order and adds `experiment_type`,
`hyperparameters` (a JSON object), `evaluation_split` and `metric_average`.
Appending to a CSV with the old header preserves its rows and leaves their
new fields blank. Omitted optional values are empty. Existing readers that
use named columns can continue to read the original fields. Use
`python -m unittest discover -s tests -v` to run the synthetic checks.

## Team

| Name | Responsibility |
|---|---|
| To be added | To be added |

## Important Notes

- Never commit API keys, passwords, access tokens, or private credentials.
- Keep virtual environments and Python cache files out of Git.
- Large datasets and model checkpoints may require Git LFS or an external storage location.
- Record dependency versions and random seeds for reproducibility.
- Do not claim performance results until they have been measured on the agreed test set.
