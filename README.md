# Credit Risk Scoring — Machine Learning App

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A portfolio-ready, bank-style machine-learning decision-support project that
predicts lower-risk (`0`) or higher-risk (`1`) credit customers.

The project uses the German Credit dataset (`credit-g`, OpenML dataset 31),
performs feature engineering, compares Logistic Regression, Decision Tree, and
Random Forest, and saves the model with the highest test ROC-AUC.

> Educational decision-support demo only. It is not a production lending
> system or an automated loan-approval tool.

## Highlights

- Cleaning, validation, and duplicate removal
- Domain-inspired feature engineering
- Reusable preprocessing and classification pipeline
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curves
- Streamlit applicant-assessment interface
- Automated unit tests

## Quick start

```powershell
git clone https://github.com/alijawad8586/codealpha-credit-risk-scoring.git
cd codealpha-credit-risk-scoring
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
py -3 train_model.py
py -3 -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Test

```powershell
py -3 -m pytest -q
```

## Project structure

```text
.
|-- app.py
|-- train_model.py
|-- test_credit_scoring.py
|-- credit_scoring_model.ipynb
|-- credit_scoring_model.pkl
|-- model_results.csv
|-- model_schema.json
|-- requirements.txt
`-- images/
```

## Responsible use

The model is trained on a public benchmark dataset and may encode historical
bias. Do not use its output for real lending, eligibility, or adverse-action
decisions. Production use would require governance, fairness evaluation,
calibration, monitoring, security controls, and human review.

## License

MIT License. See [LICENSE](LICENSE).
