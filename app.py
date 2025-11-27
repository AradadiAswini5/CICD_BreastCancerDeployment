from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "finalized_model_RandomForest.joblib")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.joblib")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.joblib")

# ---------- Load model and encoders ----------
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(LABEL_ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)

# ---------- UI Feature definitions (8 only, without Survival Months) ----------
FEATURE_COLUMNS_UI = [
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "Tumor Size (1 to 200)",
    "Progesterone Status",
    "Regional Node Examined (0 to 50)",
    "Reginol Node Positive (0 to 40)"
]

CATEGORICAL_COLUMNS = [
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "Progesterone Status",
]

NUMERIC_COLUMNS = [
    "Tumor Size (1 to 200)",
    "Regional Node Examined (0 to 50)",
    "Reginol Node Positive (0 to 40)",
]


def get_category_options():
    options = {}
    for col in CATEGORICAL_COLUMNS:
        le = label_encoders[col]
        options[col] = list(le.classes_)
    return options


@app.route("/", methods=["GET", "POST"])
def index():
    category_options = get_category_options()

    if request.method == "POST":
        errors = []
        encoded_values = []
        raw_values = {}

        # ----- 8 user inputs only -----
        for col in FEATURE_COLUMNS_UI:
            form_key = col.replace(" ", "_")
            value = request.form.get(form_key, "").strip()
            raw_values[col] = value

            if value == "":
                errors.append(f"{col} is required.")
                encoded_values.append(None)
                continue

            # Categorical
            if col in CATEGORICAL_COLUMNS:
                le = label_encoders[col]
                try:
                    encoded = le.transform([value])[0]
                    encoded_values.append(encoded)
                except:
                    errors.append(f"Invalid value for {col}.")
                    encoded_values.append(None)

            # Numeric
            else:
                try:
                    num = float(value)
                    encoded_values.append(num)
                except:
                    errors.append(f"{col} must be numeric.")
                    encoded_values.append(None)

        # ----- Add default Survival Months -----
        DEFAULT_SURVIVAL_MONTHS = 36  # neutral & safe
        encoded_values.append(DEFAULT_SURVIVAL_MONTHS)
        raw_values["Survival Months"] = DEFAULT_SURVIVAL_MONTHS

        # ----- If error, reload form -----
        if errors or any(v is None for v in encoded_values):
            return render_template(
                "index.html",
                feature_columns=FEATURE_COLUMNS_UI,
                categorical_columns=CATEGORICAL_COLUMNS,
                category_options=category_options,
                errors=errors,
                previous_values=raw_values,
            )

        # ----- Predict -----
        X_input = np.array([encoded_values])
        y_pred_encoded = model.predict(X_input)
        y_pred_label = target_encoder.inverse_transform(y_pred_encoded)[0]

        # -------- Safe UI Labels --------
        STATUS_MAPPING = {
            "Alive": "Low Risk (Favorable Prognosis)",
            "Dead": "High Risk (Requires Medical Attention)"
        }

        safe_status = STATUS_MAPPING.get(y_pred_label, "Unknown")

        return render_template(
            "result.html",
            status=safe_status,
            feature_columns=FEATURE_COLUMNS_UI,
            values=raw_values,
        )

    return render_template(
        "index.html",
        feature_columns=FEATURE_COLUMNS_UI,
        categorical_columns=CATEGORICAL_COLUMNS,
        category_options=category_options,
        errors=None,
        previous_values={},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
