"""
Rebuild the saved model on THIS computer.

When to use it
--------------
A saved model (.joblib / .pkl) can only be loaded safely with the SAME library
versions it was saved with (especially XGBoost and scikit-learn). If loading
`models/best_model.joblib` fails (for example "XGBoostError: input stream
corrupted" or an InconsistentVersionWarning), run from the project folder:

    python -m src.retrain

It does NOT repeat the long hyperparameter search. It reads the best settings that
notebook 03 found (models/model_metadata.json), trains that same model on the same
training rows (data/processed/train.csv), checks it on the same test rows, and
saves fresh .joblib and .pkl files that match the libraries installed here.

Options:  --output-dir PATH   write the files somewhere else instead of models/
"""
import argparse
import json
import pickle
import platform
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.car_features import RAW_INPUT_COLUMNS  # noqa: E402
from src.modeling import RANDOM_STATE, TARGET, build_pipeline, regression_metrics  # noqa: E402


def make_estimator(name, params):
    """Same constructors and fixed settings as notebook 03; `params` are the tuned values."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    if name == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if name.startswith("SVR"):
        return SVR(**params)
    if name.startswith("XGBoost"):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", **params)
    raise ValueError(f"Unknown model name in metadata: {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild the saved model with the libraries installed on this computer.")
    parser.add_argument("--output-dir", default=str(ROOT / "models"), help="where to write the model files (default: models/)")
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ROOT / "models" / "model_metadata.json"
    meta = json.load(open(meta_path))
    name, params = meta["selected_model"], meta["best_hyperparameters"]
    print(f"Rebuilding: {name}\nSettings:   {params}")

    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build_pipeline(make_estimator(name, params)).fit(train[RAW_INPUT_COLUMNS], train[TARGET])
        pred = model.predict(test[RAW_INPUT_COLUMNS])
    new = regression_metrics(test[TARGET], pred)
    old = meta["test_metrics"]

    print("\nTest-set check (same test rows as notebook 03):")
    print(f"  {'metric':10s}{'original run':>16s}{'rebuilt here':>16s}")
    for key in ["R2", "MAE", "RMSLE", "MedAPE_%"]:
        print(f"  {key:10s}{old[key]:16.4f}{new[key]:16.4f}")
    if abs(new["RMSLE"] - old["RMSLE"]) > 0.01:
        print("\nWARNING: the rebuilt model differs noticeably from the original run. Compare library versions.")
    else:
        print("\nThe rebuilt model matches the original run (small differences between library versions are normal).")

    joblib.dump(model, out_dir / "best_model.joblib", compress=3)
    with open(out_dir / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                "numpy": np.__version__, "joblib": joblib.__version__}
    if name.startswith("XGBoost"):
        import xgboost
        versions["xgboost"] = xgboost.__version__
    meta["library_versions"] = versions
    meta["retrained_on"] = date.today().isoformat()
    meta["retrained_test_metrics"] = {k: float(v) for k, v in new.items()}
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}: best_model.joblib, best_model.pkl, model_metadata.json")
    print("Now start the app again:  streamlit run app.py")


if __name__ == "__main__":
    main()
