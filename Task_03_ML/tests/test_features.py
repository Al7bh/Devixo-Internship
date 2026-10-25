"""Unit tests for src/car_features.py (no trained model needed)."""
import numpy as np
import pandas as pd
import pytest

from src.car_features import (REFERENCE_YEAR, CarFeatureEngineer, extract_brand)


def make_row(**overrides):
    row = {"name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500, "fuel": "Diesel",
           "transmission": "Manual", "owner": "First Owner", "mileage": "23.4 kmpl",
           "engine": "1248 CC", "max_power": "74 bhp"}
    row.update(overrides)
    return pd.DataFrame([row])


def transform(**overrides):
    return CarFeatureEngineer().fit_transform(make_row(**overrides))


def test_units_are_parsed_into_numbers():
    out = transform()
    assert out.loc[0, "mileage"] == pytest.approx(23.4)
    assert out.loc[0, "engine"] == 1248
    assert out.loc[0, "max_power"] == 74


def test_km_per_kg_is_parsed_as_a_number():
    assert transform(mileage="26.6 km/kg").loc[0, "mileage"] == pytest.approx(26.6)


def test_blank_or_zero_values_become_missing():
    assert np.isnan(transform(max_power=" bhp").loc[0, "max_power"])   # real value found in the dataset
    assert np.isnan(transform(mileage="0.0 kmpl").loc[0, "mileage"])    # zero mileage is not a measurement
    assert np.isnan(transform(engine=np.nan).loc[0, "engine"])


def test_numbers_without_units_also_work():
    """The Streamlit app / other callers may pass plain numbers."""
    out = transform(mileage=19.5, engine=1197, max_power=82.0)
    assert out.loc[0, "mileage"] == pytest.approx(19.5)
    assert out.loc[0, "engine"] == 1197


def test_car_age_and_log_km():
    out = transform(year=2014, km_driven=145500)
    assert out.loc[0, "car_age"] == REFERENCE_YEAR - 2014
    assert out.loc[0, "log_km_driven"] == pytest.approx(np.log1p(145500))


def test_car_age_never_negative():
    assert transform(year=2026).loc[0, "car_age"] == 0


def test_negative_km_becomes_missing():
    assert np.isnan(transform(km_driven=-5).loc[0, "km_driven"])


@pytest.mark.parametrize("name, expected", [
    ("Maruti Swift Dzire VDI", "Maruti"),
    ("Land Rover Discovery Sport TD4 SE", "Land Rover"),
    ("Mercedes-Benz S-Class S 350 CDI", "Mercedes-Benz"),
    ("  Hyundai i20 Asta  ", "Hyundai"),
])
def test_brand_extraction(name, expected):
    assert extract_brand(pd.Series([name])).iloc[0] == expected


def test_missing_required_column_gives_clear_error():
    with pytest.raises(ValueError, match="Missing required input columns"):
        CarFeatureEngineer().fit(make_row().drop(columns=["year"]))


def test_optional_columns_may_be_absent():
    out = transform()                      # make_row has no seats / seller_type
    assert np.isnan(out.loc[0, "seats"])
    assert "seller_type" in out.columns


def test_input_is_not_modified():
    df = make_row()
    before = df.copy()
    CarFeatureEngineer().fit_transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_row_order_and_index_are_preserved():
    df = pd.concat([make_row(year=2010), make_row(year=2015)], ignore_index=True)
    df.index = [10, 20]
    out = CarFeatureEngineer().fit_transform(df)
    assert list(out.index) == [10, 20]
    assert list(out["car_age"]) == [10, 5]
