import pytest
from pydantic import ValidationError

from serving.app import FEATURE_ORDER, MODEL_FEATURE_ORDER, Features

EXPECTED_FEATURE_ORDER = [
    "VOLTAGE",
    "CURRENT",
    "PF",
    "Temp_F",
    "Humidity",
    "WEEKEND_WEEKDAY",
    "SEASON",
    "lag1",
    "rolling_mean_3",
]

EXPECTED_MODEL_FEATURE_ORDER = [
    "VOLTAGE",
    "CURRENT",
    "PF",
    "Temp (F)",
    "Humidity (%)",
    '"WEEKEND/WEEKDAY"',
    "SEASON",
    "lag1",
    "rolling_mean_3",
]


def test_feature_order_matches_expected():
    assert FEATURE_ORDER == EXPECTED_FEATURE_ORDER


def test_model_feature_order_matches_trained_columns():
    assert MODEL_FEATURE_ORDER == EXPECTED_MODEL_FEATURE_ORDER


def test_features_model_rejects_out_of_bounds_values():
    with pytest.raises(ValidationError):
        Features(
            VOLTAGE=230.0,
            CURRENT=10.5,
            PF=1.5,
            Temp_F=82.0,
            Humidity=55.0,
            WEEKEND_WEEKDAY=0,
            SEASON=3,
            lag1=70.8,
            rolling_mean_3=72.1,
        )

    with pytest.raises(ValidationError):
        Features(
            VOLTAGE=230.0,
            CURRENT=10.5,
            PF=0.96,
            Temp_F=82.0,
            Humidity=55.0,
            WEEKEND_WEEKDAY=2,
            SEASON=3,
            lag1=70.8,
            rolling_mean_3=72.1,
        )


def test_features_model_preserves_feature_order():
    features = Features(
        VOLTAGE=230.0,
        CURRENT=10.5,
        PF=0.96,
        Temp_F=82.0,
        Humidity=55.0,
        WEEKEND_WEEKDAY=0,
        SEASON=3,
        lag1=70.8,
        rolling_mean_3=72.1,
    )

    ordered_values = [getattr(features, key) for key in FEATURE_ORDER]
    assert ordered_values == [
        230.0,
        10.5,
        0.96,
        82.0,
        55.0,
        0,
        3,
        70.8,
        72.1,
    ]
