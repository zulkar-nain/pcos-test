"""
Trains a PCOS risk-screening model using only self-reportable features
(no lab/ultrasound values), so it can power a simple web questionnaire.

Run: python train_model.py
Produces: model/pcos_model.joblib
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

DATA_PATH = os.path.join("data", "PCOS_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "pcos_model.joblib")

# Raw column names -> our clean feature names (self-reportable only)
COLUMN_MAP = {
    " Age (yrs)": "age",
    "Weight (Kg)": "weight_kg",
    "Height(Cm) ": "height_cm",
    "Cycle(R/I)": "cycle_regularity",   # 2 = Regular, 4 = Irregular
    "Cycle length(days)": "cycle_length",  # NOTE: in this dataset this is period/bleeding duration (~2-12 days), not full cycle length
    "Hip(inch)": "hip_inch",
    "Waist(inch)": "waist_inch",
    "Weight gain(Y/N)": "weight_gain",
    "hair growth(Y/N)": "hair_growth",
    "Skin darkening (Y/N)": "skin_darkening",
    "Hair loss(Y/N)": "hair_loss",
    "Pimples(Y/N)": "pimples",
    "Fast food (Y/N)": "fast_food",
    "Reg.Exercise(Y/N)": "reg_exercise",
    "PCOS (Y/N)": "pcos",
}

FEATURES = [
    "bmi",
    "waist_hip_ratio",
    "cycle_irregular",
    "cycle_length",
    "weight_gain",
    "hair_growth",
    "skin_darkening",
    "hair_loss",
    "pimples",
    "fast_food",
    "reg_exercise",
]


def load_and_engineer(path):
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    df = df[list(COLUMN_MAP.values())].copy()

    # Coerce numerics (a few cells in the source CSV have stray characters)
    for col in ["age", "weight_kg", "height_cm", "cycle_length", "hip_inch", "waist_inch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["weight_gain", "hair_growth", "skin_darkening", "hair_loss", "pimples",
                "fast_food", "reg_exercise", "cycle_regularity", "pcos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    df["waist_hip_ratio"] = df["waist_inch"] / df["hip_inch"]
    df["cycle_irregular"] = (df["cycle_regularity"] == 4).astype(int)

    X = df[FEATURES]
    y = df["pcos"].astype(int)
    return X, y


def main():
    X, y = load_and_engineer(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Export lightweight JSON weights for fast container startup without ML libs
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    import json
    weights = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0])
    }
    json_path = os.path.join(MODEL_DIR, "model_weights.json")
    with open(json_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved lightweight weights to {json_path}")


if __name__ == "__main__":
    main()
