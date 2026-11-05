"""Inference utilities for trained HAR models.

TODO: implement checkpoint loading and input preprocessing consistently with
the training pipeline.
"""


def load_trained_model(checkpoint_path, **kwargs):
    """Load and return a trained model from a checkpoint."""
    raise NotImplementedError("Implement checkpoint loading after model selection.")


def predict(model, samples, **kwargs):
    """Predict activity labels for one or more samples."""
    raise NotImplementedError("Implement prediction for the chosen input format.")
