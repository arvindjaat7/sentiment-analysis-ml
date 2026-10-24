from flask import Flask, render_template, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model/sentiment_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not found. Run train_model.py first."
        }), 500

    data = request.get_json(silent=True) or request.form
    review = str(data.get("review", "")).strip()

    if not review:
        return jsonify({
            "error": "Please enter a review."
        }), 400

    prediction = model.predict([review])[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([review])[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return jsonify({
        "sentiment": str(prediction).capitalize(),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)