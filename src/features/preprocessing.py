from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureSpec:
    target: str = "is_fraud"
    time_col: str = "timestamp"


def build_preprocessor(df: pd.DataFrame) -> Tuple[Pipeline, List[str], List[str]]:
    """
    Build a preprocessing pipeline:
    - numeric: median impute
    - categorical: most_frequent impute + one-hot
    Returns pipeline and lists of numeric/categorical columns used.
    """
    drop_cols = {"transaction_id", "customer_id", "card_id", "merchant_id", "mcc_desc"}  # IDs not learned directly
    target = FeatureSpec().target
    time_col = FeatureSpec().time_col

    feature_cols = [c for c in df.columns if c not in drop_cols and c not in {target, time_col}]
    # split by dtype
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)]), num_cols, cat_cols