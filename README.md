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
