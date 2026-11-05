"""Data loading and preprocessing utilities for the HAR project.

TODO: implement these functions after the dataset format and feature schema
have been confirmed.
"""


def load_data(data_dir):
    """Load raw samples and labels from the supplied directory."""
    raise NotImplementedError("Define the dataset format before implementing data loading.")


def preprocess_data(features, labels=None, fit=False, preprocessor=None):
    """Fit/apply preprocessing and return transformed data."""
    raise NotImplementedError("Choose preprocessing steps based on the dataset.")
