from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import os
import json

# Only import gdown when needed
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

from sklearn.preprocessing import LabelEncoder

def sanitize_feature_name(name):
    name = str(name)
    name = re.sub(r'[\[\]{}:",.<>\s]', '_', name)
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if not name:
        return "unknown_feature"
    return name

MODEL_PATH = "random_forest_best_model.joblib"
DIET_PATH = "diseases_diets.csv"
MED_PATH = "diseases_medications.csv"
SYMPTOM_FEATURES_PATH = "symptom_features.json"

# ---- DOWNLOAD MODEL IF NEEDED ----
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file from Google Drive...")
        url = "https://drive.google.com/uc?id=1ZjZX5Bw56NsgKfNc3tXrmr39qjnpzMej"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")

download_model()

# ---- CONTINUE WITH NORMAL LOADING ----
model = joblib.load(MODEL_PATH)

# Load LabelEncoder classes
disease_classes = []
if os.path.exists(MED_PATH):
    df_med = pd.read_csv(MED_PATH)
    disease_classes = [d.lower().strip() for d in df_med['Disease'].unique()]
else:
    raise Exception("Cannot find diseases_medications.csv to get disease classes.")

le = LabelEncoder()
le.fit(disease_classes)

def load_map(path, value_col):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if 'Disease' in df.columns and value_col in df.columns:
        df['Disease'] = df['Disease'].str.lower().str.strip()
        return df.set_index('Disease')[value_col].to_dict()
    return {}

diet_map = load_map(DIET_PATH, "Diets")
med_map = load_map(MED_PATH, "Medication")

if os.path.exists(SYMPTOM_FEATURES_PATH):
    with open(SYMPTOM_FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    all_feature_names_sanitized = [sanitize_feature_name(s) for s in feature_names]
else:
    raise Exception("symptom_features.json with your true feature names is missing!")

def predict(symptoms):
    x = pd.DataFrame(np.zeros((1, len(all_feature_names_sanitized))), columns=all_feature_names_sanitized)
    recognized = False
    for orig in symptoms:
        sanitized = sanitize_feature_name(orig)
        if sanitized in x.columns:
            x.loc[0, sanitized] = 1
            recognized = True
    if not recognized:
        return {"error": "No symptoms matched model features."}

    encoded_pred = model.predict(x)[0]
    pred_disease = le.inverse_transform([encoded_pred])[0]
    pred_disease_lower = pred_disease.lower().strip()

    # Diets
    diet_raw = diet_map.get(pred_disease_lower, None)
    if diet_raw:
        diet_points = [d.strip() for d in re.split(r'\.\s*(?=[A-Z])', diet_raw) if d.strip()]
        diets = ["- " + (d if d.endswith('.') else d+'.') for d in diet_points] if diet_points else []
    else:
        diets = ["- Diet information not available for this disease."]

    # Medications
    med_raw = med_map.get(pred_disease_lower, None)
    if med_raw:
        med_points = [m.strip() for m in re.split(r',\s*', med_raw) if m.strip()]
        medications = ["- " + m for m in med_points] if med_points else []
    else:
        medications = ["- Medication information not available for this disease."]

    return {
        "predicted_disease": pred_disease,
        "diets": diets,
        "medications": medications
    }

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please send a JSON with a 'symptoms' field (list of symptoms)"}), 400
    symptoms = data["symptoms"]
    result = predict(symptoms)
    return jsonify(result)

@app.route("/", methods=["GET"])
def root():
    return "API is up. POST to /recommend with {'symptoms': [list of symptoms]}"

if __name__ == "__main__":
    app.run(debug=True)
