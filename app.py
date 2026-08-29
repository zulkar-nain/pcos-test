"""
PCOS Risk Screening web app.

Presents a 3-step questionnaire (physical measurements, menstrual history,
symptoms & lifestyle) and returns an estimated PCOS risk score using a
logistic-regression model trained on self-reportable features only.

This is an educational screening tool, NOT a medical diagnosis.
"""
import os
import secrets
import joblib
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "pcos_model.joblib")
_bundle = joblib.load(MODEL_PATH)
PIPELINE = _bundle["pipeline"]
FEATURES = _bundle["features"]


def yn(value):
    """Convert a 'yes'/'no' form value to 1/0."""
    return 1 if value == "yes" else 0


@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/questions/1", methods=["GET", "POST"])
def step1():
    """Set 1: Physical measurements."""
    if request.method == "POST":
        session["age"] = float(request.form["age"])
        session["weight_kg"] = float(request.form["weight_kg"])
        session["height_cm"] = float(request.form["height_cm"])
        session["waist_inch"] = float(request.form["waist_inch"])
        session["hip_inch"] = float(request.form["hip_inch"])
        return redirect(url_for("step2"))
    return render_template("step1.html")


@app.route("/questions/2", methods=["GET", "POST"])
def step2():
    """Set 2: Menstrual history."""
    if "age" not in session:
        return redirect(url_for("step1"))
    if request.method == "POST":
        session["cycle_irregular"] = 1 if request.form["cycle_regularity"] == "irregular" else 0
        session["cycle_length"] = float(request.form["cycle_length"])
        return redirect(url_for("step3"))
    return render_template("step2.html")


@app.route("/questions/3", methods=["GET", "POST"])
def step3():
    """Set 3: Symptoms & lifestyle."""
    if "cycle_length" not in session:
        return redirect(url_for("step1"))
    if request.method == "POST":
        session["weight_gain"] = yn(request.form.get("weight_gain"))
        session["hair_growth"] = yn(request.form.get("hair_growth"))
        session["skin_darkening"] = yn(request.form.get("skin_darkening"))
        session["hair_loss"] = yn(request.form.get("hair_loss"))
        session["pimples"] = yn(request.form.get("pimples"))
        session["fast_food"] = yn(request.form.get("fast_food"))
        session["reg_exercise"] = yn(request.form.get("reg_exercise"))
        return redirect(url_for("result"))
    return render_template("step3.html")


@app.route("/result")
def result():
    required = ["age", "weight_kg", "height_cm", "waist_inch", "hip_inch",
                "cycle_irregular", "cycle_length", "weight_gain", "hair_growth",
                "skin_darkening", "hair_loss", "pimples", "fast_food", "reg_exercise"]
    if not all(k in session for k in required):
        return redirect(url_for("index"))

    bmi = session["weight_kg"] / ((session["height_cm"] / 100) ** 2)
    waist_hip_ratio = session["waist_inch"] / session["hip_inch"]

    row = {
        "bmi": bmi,
        "waist_hip_ratio": waist_hip_ratio,
        "cycle_irregular": session["cycle_irregular"],
        "cycle_length": session["cycle_length"],
        "weight_gain": session["weight_gain"],
        "hair_growth": session["hair_growth"],
        "skin_darkening": session["skin_darkening"],
        "hair_loss": session["hair_loss"],
        "pimples": session["pimples"],
        "fast_food": session["fast_food"],
        "reg_exercise": session["reg_exercise"],
    }
    X = pd.DataFrame([row], columns=FEATURES)
    probability = float(PIPELINE.predict_proba(X)[0, 1])
    risk_percent = round(probability * 100, 1)

    if risk_percent >= 65:
        risk_level, risk_class = "High", "risk-high"
    elif risk_percent >= 35:
        risk_level, risk_class = "Moderate", "risk-moderate"
    else:
        risk_level, risk_class = "Low", "risk-low"

    return render_template(
        "result.html",
        risk_percent=risk_percent,
        risk_level=risk_level,
        risk_class=risk_class,
        bmi=round(bmi, 1),
        waist_hip_ratio=round(waist_hip_ratio, 2),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
