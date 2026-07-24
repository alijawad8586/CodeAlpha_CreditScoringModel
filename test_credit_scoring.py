import pandas as pd

from train_model import engineer_features, model_candidates


def test_engineer_features_creates_expected_columns():
    frame = pd.DataFrame(
        {"credit_amount": [120000], "duration": [12], "age": [30], "existing_credits": [2]}
    )
    result = engineer_features(frame)
    assert result.loc[0, "credit_amount_per_month"] == 10000
    assert result.loc[0, "credit_per_existing_account"] == 60000
    assert str(result.loc[0, "age_group"]) == "young_adult"


def test_engineer_features_handles_zero_denominators():
    frame = pd.DataFrame(
        {"credit_amount": [1000], "duration": [0], "existing_credits": [0]}
    )
    result = engineer_features(frame)
    assert result.loc[0, "credit_amount_per_month"] == 1000
    assert result.loc[0, "credit_per_existing_account"] == 1000


def test_all_required_models_are_present():
    assert set(model_candidates()) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    }
