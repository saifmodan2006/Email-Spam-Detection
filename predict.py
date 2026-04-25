from __future__ import annotations

import argparse

from spam_detector.prediction import SpamDetectorService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict whether a message is spam.")
    parser.add_argument("text", type=str, help="Message text to classify.")
    parser.add_argument("--artifact", type=str, default="artifacts/spam_model.joblib", help="Model artifact path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = SpamDetectorService(args.artifact)
    result = service.predict_text(args.text)
    print(f"label={result.label}")
    print(f"spam_probability={result.spam_probability:.4f}")
    print(f"ham_probability={result.ham_probability:.4f}")


if __name__ == "__main__":
    main()
