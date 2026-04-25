# Inbox Shield: Email Spam Detection

Inbox Shield is a complete machine learning project that detects spam-like email or SMS messages.

It includes:
- A reusable training pipeline with scikit-learn
- Saved model artifacts for fast repeated inference
- CLI tools for training and prediction
- A polished Streamlit dashboard for interactive testing

This repository is designed to be simple to run, easy to explain in interviews, and practical to extend.

## Features

- Binary classification: spam vs ham
- TF-IDF + Logistic Regression baseline model
- Train/test split with evaluation metrics
- Saved model artifact at artifacts/spam_model.joblib
- Interactive dashboard with probability breakdown
- Retraining from sample dataset or custom CSV upload

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## Project Structure

- app.py: Streamlit UI
- train.py: model training CLI
- predict.py: single-text prediction CLI
- spam_detector/model.py: model wrapper and artifact IO
- spam_detector/training.py: dataset loading, training, evaluation
- spam_detector/prediction.py: high-level prediction service
- data/sample_emails.csv: starter labeled dataset

## Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train a Model

Train with bundled sample data:

```powershell
python train.py --data data/sample_emails.csv
```

Train with your own dataset:

```powershell
python train.py --data path\to\your_dataset.csv
```

Required CSV columns:
- label: spam or ham
- text: raw message content

## Run Prediction from CLI

```powershell
python predict.py "Congratulations, you won a free prize! Click now!"
```

Example output:

```text
label=spam
spam_probability=0.5872
ham_probability=0.4128
```

## Run the Web App

```powershell
streamlit run app.py
```

The UI supports:
- Manual message testing
- One-click retraining on sample data
- CSV upload retraining
- Model quality metrics and confusion matrix preview

## Evaluation Metrics

The training script prints:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Classification report

## Common Commands

```powershell
# Train with default sample
python train.py --data data/sample_emails.csv

# Save to a custom artifact path
python train.py --data data/sample_emails.csv --artifact artifacts/my_model.joblib

# Predict with a custom artifact
python predict.py "Urgent! Verify your account now" --artifact artifacts/my_model.joblib
```

## Notes for Production Use

- The bundled dataset is intentionally small for quick smoke tests.
- For real performance, use a larger and cleaner labeled dataset.
- Add cross-validation, threshold tuning, and monitoring before deployment.

## License

Use this repository for learning and portfolio projects. Add your preferred license before public release.
