from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import os
import json
import psutil

# Import gdown only when needed
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

def get_ram(label=""):
    ram = psutil.virtual_memory().used / (1024*1024)
    print(f"RAM {label}: {ram:.2f} MB", flush=True)
    return ram

MODEL_PATH = "random_forest_new.joblib"
DIET_PATH = "diseases_diets.csv"
MED_PATH = "diseases_medications.csv"
SYMPTOM_FEATURES_PATH = "symptom_features.json"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file from Google Drive...", flush=True)
        url = "https://drive.google.com/uc?id=1HbiT4SN1hFkLEGolizmmd5bQqPOTM1wa"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.", flush=True)
    else:
        print("Model file already exists, skipping download.", flush=True)

# ---- STARTUP LOGS ----
get_ram("before model download")
download_model()
get_ram("before model load")

try:
    print("Loading model...", flush=True)
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print(f"ERROR during model load: {e}", flush=True)
    raise

get_ram("after model load")

# ---- LABEL ENCODER ----
print("Loading disease classes...", flush=True)
disease_classes = []
if os.path.exists(MED_PATH):
    df_med = pd.read_csv(MED_PATH)
    disease_classes = [d.lower().strip() for d in df_med['Disease'].unique()]
else:
    raise Exception("Cannot find diseases_medications.csv to get disease classes.")

le = LabelEncoder()
le.fit(disease_classes)
print(f"Loaded {len(disease_classes)} disease classes.", flush=True)

# ---- DIET & MED ----
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

# ---- SYMPTOM FEATURES ----
if os.path.exists(SYMPTOM_FEATURES_PATH):
    with open(SYMPTOM_FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    all_feature_names_sanitized = [sanitize_feature_name(s) for s in feature_names]
    print(f"Loaded {len(all_feature_names_sanitized)} symptom features.", flush=True)
else:
    raise Exception("symptom_features.json with your true feature names is missing!")

# ---- PREDICTION LOGIC ----
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

# ---- FLASK ----
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def predict_api():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please send a JSON with a 'symptoms' field (list of symptoms)"}), 400
    symptoms = data["symptoms"]
    print(f"Received symptoms: {symptoms}", flush=True)
    result = predict(symptoms)
    print(f"Prediction result: {result}", flush=True)
    return jsonify(result)

@app.route("/", methods=["GET"])
def root():
    return "API is up. POST to /recommend with {'symptoms': [list of symptoms]}"

if __name__ == "__main__":
    app.run(debug=True)
