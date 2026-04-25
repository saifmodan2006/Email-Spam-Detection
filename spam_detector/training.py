from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .model import SpamModel, save_model


DEFAULT_DATASET = [
    ("ham", "Hey, are we still on for lunch today?"),
    ("ham", "Please review the project notes when you have a moment."),
    ("ham", "I will send the meeting agenda before noon."),
    ("ham", "Your package was delivered to the front desk."),
    ("ham", "Can you call me back when you are free?"),
    ("ham", "The weekly report is attached for your review."),
    ("ham", "Thanks for your help with the presentation."),
    ("ham", "I booked the conference room for tomorrow at 10."),
    ("ham", "Let us catch up after work this evening."),
    ("ham", "Happy birthday! Hope you have a great day."),
    ("ham", "The invoice has been approved and sent to finance."),
    ("ham", "I updated the spreadsheet with the latest figures."),
    ("ham", "Your flight is confirmed for next Tuesday."),
    ("ham", "Lunch was great, we should do that again soon."),
    ("ham", "Reminder: dentist appointment at 3 PM tomorrow."),
    ("spam", "Congratulations, you have won a free prize. Click now!"),
    ("spam", "Urgent! Your account is locked. Verify your password immediately."),
    ("spam", "You have been selected for a cash reward. Reply to claim."),
    ("spam", "Claim your free vacation ticket today by calling this number."),
    ("spam", "Limited-time offer: buy now and get 90 percent off."),
    ("spam", "Your phone has won a gift card. Enter the code to redeem."),
    ("spam", "Act fast, your subscription will expire unless you confirm payment."),
    ("spam", "Win big prizes now by joining this exclusive contest."),
    ("spam", "We noticed suspicious activity. Verify your banking details immediately."),
    ("spam", "Earn money from home with this simple guaranteed method."),
    ("spam", "Free access unlocked. Download the attachment to continue."),
    ("spam", "You are pre-approved for an instant loan. Apply today."),
    ("spam", "Click the link to claim your compensation reward."),
    ("spam", "This is your final notice to activate the bonus now."),
    ("spam", "Important: your lottery entry has already been selected."),
]


@dataclass(slots=True)
class TrainingResult:
    model: SpamModel
    metrics: dict[str, Any]
    evaluation_rows: list[dict[str, Any]]


def build_default_dataset() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_DATASET, columns=["label", "text"])


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        return build_default_dataset()

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    normalized_columns = {column.lower(): column for column in frame.columns}
    if "label" not in normalized_columns or "text" not in normalized_columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")

    frame = frame.rename(columns={normalized_columns["label"]: "label", normalized_columns["text"]: "text"})
    return frame[["label", "text"]].dropna().reset_index(drop=True)


def _normalize_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"spam": 1, "ham": 0, "0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}
    if set(normalized.unique()) <= set(mapping):
        return normalized.map(mapping)
    if normalized.isin(["spam", "ham"]).all():
        return normalized.map({"ham": 0, "spam": 1})
    raise ValueError("Labels must be spam/ham or binary equivalents.")


def train_and_evaluate(dataset: pd.DataFrame, random_state: int = 42, artifact_path: str | Path | None = None) -> TrainingResult:
    working = dataset.copy()
    working["text"] = working["text"].astype(str).fillna("")
    working["target"] = _normalize_labels(working["label"])

    train_frame, test_frame = train_test_split(
        working,
        test_size=0.25,
        random_state=random_state,
        stratify=working["target"],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(train_frame["text"], train_frame["target"])

    predictions = pipeline.predict(test_frame["text"])
    metrics = {
        "accuracy": float(accuracy_score(test_frame["target"], predictions)),
        "precision": float(precision_score(test_frame["target"], predictions, zero_division=0)),
        "recall": float(recall_score(test_frame["target"], predictions, zero_division=0)),
        "f1": float(f1_score(test_frame["target"], predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(test_frame["target"], predictions).tolist(),
        "classification_report": classification_report(test_frame["target"], predictions, target_names=["ham", "spam"], zero_division=0),
        "train_size": int(len(train_frame)),
        "test_size": int(len(test_frame)),
    }

    model = SpamModel(
        pipeline=pipeline,
        metadata={
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state,
            "rows": int(len(working)),
        },
    )

    if artifact_path is not None:
        save_model(model, artifact_path)

    evaluation_rows = [
        {
            "text": text,
            "actual": "spam" if actual == 1 else "ham",
            "predicted": "spam" if predicted == 1 else "ham",
        }
        for text, actual, predicted in zip(test_frame["text"], test_frame["target"], predictions, strict=False)
    ]

    return TrainingResult(model=model, metrics=metrics, evaluation_rows=evaluation_rows)


def load_training_artifact(artifact_path: str | Path) -> SpamModel:
    from .model import load_model

    return load_model(artifact_path)
