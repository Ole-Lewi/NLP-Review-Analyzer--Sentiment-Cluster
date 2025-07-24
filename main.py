from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load model and vectorizer
model = joblib.load("kmeans_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_cluster(review: Review):
    vec = vectorizer.transform([review.text])
    cluster = model.predict(vec)[0]
    return {"cluster": int(cluster)}
