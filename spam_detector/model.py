from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(slots=True)
class SpamModel:
    pipeline: Any
    metadata: dict[str, Any]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)


def save_model(model: SpamModel, artifact_path: str | Path) -> Path:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model.pipeline, "metadata": model.metadata}, path)
    return path


def load_model(artifact_path: str | Path) -> SpamModel:
    payload = joblib.load(Path(artifact_path))
    return SpamModel(pipeline=payload["pipeline"], metadata=payload["metadata"])
