"""
train_model.py
--------------
Loads cardekho.csv, cleans the data, trains a Random Forest regressor
inside a scikit-learn Pipeline, and saves the artefact to model.pkl.

Run:  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv("cardekho.csv")
print(f"Raw dataset shape: {df.shape}")

# ── 2. Clean / feature engineering ──────────────────────────────────────────

def strip_unit(series):
    """Extract leading numeric value from strings like '23.4 kmpl' or '1248 CC'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)")[0],
        errors="coerce"
    )

# mileage / engine / max_power may have been read as strings in some versions
for col in ["mileage(km/ltr/kg)", "engine", "max_power"]:
    if df[col].dtype == object:
        df[col] = strip_unit(df[col])

# Car age is more informative than raw year
df["car_age"] = 2024 - df["year"]

# Drop rows with missing target or critical features
df.dropna(subset=["selling_price", "mileage(km/ltr/kg)", "engine",
                   "max_power", "seats"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Cleaned dataset shape: {df.shape}")

# ── 3. Feature / target split ────────────────────────────────────────────────
FEATURES = ["car_age", "km_driven", "mileage(km/ltr/kg)", "engine",
            "max_power", "seats", "fuel", "seller_type",
            "transmission", "owner"]
TARGET = "selling_price"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ── 4. Column groups ─────────────────────────────────────────────────────────
numeric_cols = ["car_age", "km_driven", "mileage(km/ltr/kg)",
                "engine", "max_power", "seats"]
categorical_cols = ["fuel", "seller_type", "transmission", "owner"]

# ── 5. Pre-processor ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1), categorical_cols),
])

# ── 6. Pipeline ───────────────────────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )),
])

# ── 7. Train / evaluate ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\nTest MAE : INR {mae:,.0f}")
print(f"Test R2  : {r2:.4f}")

# ── 8. Metadata stored alongside the model (for the UI dropdowns) ─────────────
meta = {
    "features": FEATURES,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "unique_fuels":        sorted(df["fuel"].dropna().unique().tolist()),
    "unique_seller_types": sorted(df["seller_type"].dropna().unique().tolist()),
    "unique_transmissions":sorted(df["transmission"].dropna().unique().tolist()),
    "unique_owners":       sorted(df["owner"].dropna().unique().tolist()),
    "min_year": int(df["year"].min()),
    "max_year": int(df["year"].max()),
    "mae": round(mae, 2),
    "r2":  round(r2, 4),
}

# ── 9. Save ───────────────────────────────────────────────────────────────────
joblib.dump({"pipeline": pipeline, "meta": meta}, "model.pkl")
print("\nSaved -> model.pkl")
