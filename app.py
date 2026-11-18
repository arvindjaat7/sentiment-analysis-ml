
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model/sentiment_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("review", "").strip()

        if not text:
            return jsonify({"error": "Please enter some text."}), 400

        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        percentages = {
            str(label): round(float(probability) * 100, 2)
            for label, probability in zip(classes, probabilities)
        }

        predicted_sentiment = str(classes[probabilities.argmax()])

        return jsonify({
            "sentiment": predicted_sentiment,
            "percentages": percentages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
