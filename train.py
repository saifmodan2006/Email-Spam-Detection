from __future__ import annotations

import argparse
import json
from pathlib import Path

from spam_detector.training import load_dataset, train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the email spam detector.")
    parser.add_argument("--data", type=str, default=None, help="Path to a CSV with label,text columns.")
    parser.add_argument("--artifact", type=str, default="artifacts/spam_model.joblib", help="Where to save the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/test split.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset(args.data)
    result = train_and_evaluate(dataset, random_state=args.seed, artifact_path=args.artifact)

    print(json.dumps(result.metrics, indent=2))
    print(f"Saved model to {Path(args.artifact).resolve()}")


if __name__ == "__main__":
    main()
