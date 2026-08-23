# PCOS Risk Screening Web App

A Flask web app that estimates PCOS (Polycystic Ovary Syndrome) risk from a
short, 3-part questionnaire. The scoring model is a logistic regression
trained on the [PCOS-Risk-Predictor-Identifying-At-Risk-Individuals](https://github.com/zulkar-nain/PCOS-Risk-Predictor-Identifying-At-Risk-Individuals)
dataset, restricted to features a person can self-report (no lab tests or
ultrasound values).

## Questionnaire

1. **Physical measurements** – age, weight, height, waist & hip circumference
   (used to compute BMI and waist:hip ratio).
2. **Menstrual history** – cycle regularity and period duration.
3. **Symptoms & lifestyle** – weight gain, excess hair growth, skin
   darkening, hair loss, acne, fast food habits, exercise habits.

The result page shows an estimated risk percentage and Low / Moderate / High
risk label.

## Setup

```powershell
pip install -r requirements.txt
python train_model.py   # retrains model/pcos_model.joblib from data/PCOS_data.csv
python app.py            # runs the dev server on http://127.0.0.1:5000
```

## Project structure

```
app.py              Flask routes and inference logic
train_model.py       Trains the logistic regression model from the dataset
data/PCOS_data.csv    Source dataset
model/pcos_model.joblib  Saved trained model + feature list
templates/            Jinja2 HTML templates (one per questionnaire step)
static/style.css       App styling
```

## Disclaimer

This tool is for educational/screening purposes only and does not provide a
medical diagnosis. Anyone concerned about PCOS symptoms should consult a
qualified healthcare provider.
