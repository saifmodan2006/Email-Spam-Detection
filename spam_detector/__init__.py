"""Email spam detection package."""

from .model import SpamModel, load_model, save_model
from .training import build_default_dataset, train_and_evaluate

__all__ = ["SpamModel", "load_model", "save_model", "build_default_dataset", "train_and_evaluate"]
