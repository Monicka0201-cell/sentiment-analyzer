from flask import Flask, render_template, request, redirect
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis")

# MongoDB connection (local)
client = MongoClient("mongodb://localhost:27017/")
db = client["sentimentDB"]
collection = db["reviews"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["review"]

        # Predict sentiment
        result = sentiment_model(text)[0]

        sentiment = result["label"]
        score = result["score"]

        # Save to MongoDB
        data = {
            "review": text,
            "sentiment": sentiment,
            "score": score,
            "date": datetime.now()
        }
        collection.insert_one(data)

        return render_template("index.html", result=sentiment, score=round(score, 2))

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    data = list(collection.find())

    positive = sum(1 for d in data if d["sentiment"] == "POSITIVE")
    negative = sum(1 for d in data if d["sentiment"] == "NEGATIVE")
    neutral = len(data) - (positive + negative)

    return render_template(
        "dashboard.html",
        positive=positive,
        negative=negative,
        neutral=neutral,
        total=len(data)
    )

if __name__ == "__main__":
    app.run(debug=True)
