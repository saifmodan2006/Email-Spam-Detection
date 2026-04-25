from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import SpamModel, load_model
from .training import build_default_dataset, train_and_evaluate


@dataclass(slots=True)
class PredictionResult:
    label: str
    spam_probability: float
    ham_probability: float


class SpamDetectorService:
    def __init__(self, artifact_path: str | Path = "artifacts/spam_model.joblib") -> None:
        self.artifact_path = Path(artifact_path)
        self._model: SpamModel | None = None

    @property
    def model(self) -> SpamModel:
        if self._model is None:
            if self.artifact_path.exists():
                self._model = load_model(self.artifact_path)
            else:
                training_result = train_and_evaluate(build_default_dataset(), artifact_path=self.artifact_path)
                self._model = training_result.model
        return self._model

    def predict_text(self, text: str) -> PredictionResult:
        probabilities = self.model.predict_proba([text])[0]
        spam_probability = float(probabilities[1])
        ham_probability = float(probabilities[0])
        label = "spam" if spam_probability >= 0.5 else "ham"
        return PredictionResult(label=label, spam_probability=spam_probability, ham_probability=ham_probability)
