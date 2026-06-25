import pandas as pd
import pytest

from stroke import config, data


def test_load_dataset_stratified_split():
    ds = data.load_dataset(test_size=0.2, random_state=0)
    # Positive rate should be preserved across train/valid (stratified).
    p_train = ds.y_train.mean()
    p_valid = ds.y_valid.mean()
    assert abs(p_train - p_valid) < 0.01
    # No overlap in indices is guaranteed by train_test_split; check shapes.
    assert len(ds.X_train) > len(ds.X_valid)
    assert list(ds.X_train.columns) == config.ALL_RAW_FEATURES


def test_class_balance_keys():
    ds = data.load_dataset()
    bal = data.class_balance(ds.y_train)
    assert set(bal) == {
        "n_total",
        "n_positive",
        "n_negative",
        "positive_rate",
        "imbalance_ratio",
    }
    assert 0 < bal["positive_rate"] < 0.2  # rare positive class


def test_age_group_bins():
    df = pd.DataFrame({"age": [5, 25, 70]})
    out = data.add_age_group(df)
    assert list(out["age_group"].astype(str)) == ["<18", "18-34", "65-79"]
