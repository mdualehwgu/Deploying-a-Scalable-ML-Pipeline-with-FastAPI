import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


@pytest.fixture
def sample_data():
    """Fixture that returns a small sample DataFrame for testing."""
    data = pd.DataFrame({
        "age": [39, 50, 38, 53, 28],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721, 338409],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th", "Bachelors"],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Married-civ-spouse",
            "Married-civ-spouse",
        ],
        "occupation": [
            "Adm-clerical",
            "Exec-managerial",
            "Handlers-cleaners",
            "Handlers-cleaners",
            "Prof-specialty",
        ],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband", "Wife"],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "Cuba",
        ],
        "salary": ["<=50K", "<=50K", "<=50K", ">50K", "<=50K"],
    })
    return data


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_train_model_returns_classifier(sample_data, cat_features):
    """
    Test that train_model returns a RandomForestClassifier fitted on the data.
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_"), "Model should be fitted"


def test_compute_model_metrics_returns_expected_values(sample_data, cat_features):
    """
    Test that compute_model_metrics returns precision, recall, and F1 all in [0, 1].
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0.0 <= precision <= 1.0, "Precision should be between 0 and 1"
    assert 0.0 <= recall <= 1.0, "Recall should be between 0 and 1"
    assert 0.0 <= fbeta <= 1.0, "F1 should be between 0 and 1"


def test_inference_output_shape(sample_data, cat_features):
    """
    Test that inference returns an array with the same length as the input.
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert preds.shape[0] == X.shape[0], (
        "Predictions length should match number of input samples"
    )
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"


def test_apply_label():
    """
    Test that apply_label converts binary predictions to string salary labels.
    """
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


def test_process_data_shapes(sample_data, cat_features):
    """
    Test that process_data returns arrays with consistent shapes.
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    assert X.shape[0] == len(sample_data), "X rows should match input rows"
    assert y.shape[0] == len(sample_data), "y length should match input rows"
    assert X.shape[1] > 0, "X should have at least one feature column"
