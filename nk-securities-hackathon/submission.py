"""
Python script used for NK Securities submission

Author: Harsh Parikh
Date: 8th June 2025
"""

import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Importing the files
training_data = pd.read_parquet("/kaggle/input/nk-iv-prediction/train_data.parquet")
test_data = pd.read_parquet("/kaggle/input/nk-iv-prediction/test_data.parquet")

copy_test_data = test_data.copy()

iv_columns = copy_test_data.columns[2:54]  # These columns have the IVs
feature_columns = ["underlying"] + [
    f"X{i}" for i in range(42)
]  # Fetching out the features

# Pre-imputing all IVs to use them as input features
iv_imputer = SimpleImputer(strategy="mean")
iv_imputed = pd.DataFrame(
    iv_imputer.fit_transform(copy_test_data[iv_columns]), columns=iv_columns
)

for target_col in iv_columns:
    # Creating a missing value mask
    missing_mask = copy_test_data[target_col].isna()
    known_mask = ~missing_mask

    # Defining the training data
    X_train = pd.concat(
        [
            copy_test_data.loc[known_mask, feature_columns].reset_index(drop=True),
            iv_imputed.loc[known_mask, iv_columns.drop(target_col)].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    y_train = copy_test_data.loc[known_mask, target_col].values

    # Defining the stacked regressor
    base_models = [
        ("ridge", Ridge(alpha=1.0)),
        (
            "xgb",
            XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05, verbosity=0
            ),
        ),
    ]
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=XGBRegressor(n_estimators=50, verbosity=0),
    )
    stack.fit(X_train, y_train)

    # Defining the prediction set
    X_pred = pd.concat(
        [
            copy_test_data.loc[missing_mask, feature_columns].reset_index(drop=True),
            iv_imputed.loc[missing_mask, iv_columns.drop(target_col)].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # Predicting and updating the values
    preds = stack.predict(X_pred)
    copy_test_data.loc[missing_mask, target_col] = np.clip(preds, 0.05, 2.5)

# Matching the order of submission
submission_data = copy_test_data[test_data.columns[:54]].drop(["underlying"], axis=1)
submission_data.to_csv("local_poly_submission.csv", index=False)
