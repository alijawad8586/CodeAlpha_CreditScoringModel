"""Generate a compact, runnable notebook from the maintained training module."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Credit Scoring Model\n"
        "German Credit classification with feature engineering, three models, "
        "evaluation, artifact saving, and a Streamlit deployment."
    ),
    nbf.v4.new_code_cell(
        "from train_model import load_data\n"
        "X, y, raw_columns = load_data()\n"
        "print('Features:', X.shape)\n"
        "print(y.value_counts())\n"
        "X.head()"
    ),
    nbf.v4.new_markdown_cell(
        "## Engineered features\n"
        "`load_data()` adds monthly credit burden, age group, and credit per "
        "existing account while preserving the original app-input columns."
    ),
    nbf.v4.new_code_cell(
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.20, random_state=42, stratify=y\n"
        ")\n"
        "print(X_train.shape, X_test.shape)\n"
        "print(y_train.value_counts())"
    ),
    nbf.v4.new_markdown_cell(
        "## Train, compare, and save\n"
        "The shared module fits Logistic Regression, Decision Tree, and Random "
        "Forest and selects the highest ROC-AUC model."
    ),
    nbf.v4.new_code_cell(
        "from train_model import train_and_save\n"
        "results = train_and_save()\n"
        "results"
    ),
    nbf.v4.new_markdown_cell(
        "## Prediction\n"
        "The saved object is a complete preprocessing and classifier pipeline."
    ),
    nbf.v4.new_code_cell(
        "import joblib\n"
        "model = joblib.load('credit_scoring_model.pkl')\n"
        "sample = X_test.iloc[[0]]\n"
        "probability = model.predict_proba(sample)[0, 1]\n"
        "{'prediction': 'High Risk' if model.predict(sample)[0] else 'Low Risk', "
        "'risk_probability': round(probability * 100, 2)}"
    ),
]
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nbf.write(nb, "credit_scoring_model.ipynb")
print("Created credit_scoring_model.ipynb")
