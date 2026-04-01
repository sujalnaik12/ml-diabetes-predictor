import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form[f]) for f in [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1] * 100

    result = f"⚠️ High Risk of Diabetes ({probability:.1f}% probability)" if prediction == 1 else f"✅ Low Risk of Diabetes ({probability:.1f}% probability)"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)