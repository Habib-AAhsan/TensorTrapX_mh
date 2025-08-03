from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

import redis
from rq import Queue
from feedback_worker import save_feedback_job



from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server

# Start a separate metrics server (optional, if not using FastAPI route)
# start_http_server(9000)  # Alternative: expose on own port

# Prometheus metrics
prediction_count = Counter("prediction_count", "Total number of predictions")
prediction_confidence = Gauge("prediction_confidence", "Confidence of the last prediction")
prediction_label_total = Counter("prediction_label_total", "Count of predictions per label", ["label"])


# Load the model
model_path = "model/best_model.keras"
model = tf.keras.models.load_model(model_path)

# Class names
class_names = ["Benign", "Malignant"]

# FastAPI instance
app = FastAPI(
    title="TensorTrapX (mh) – Breast Cancer Predictor API",
    description="Real-time prediction using trained TensorFlow model",
    version="1.0"
)

# ✅ Redis connection setup
redis_conn = redis.Redis(host="localhost", port=6379, db=0)
feedback_queue = Queue("feedback", connection=redis_conn)

# ✅ Pydantic model for feedback
class FeedbackInput(BaseModel):
    user_id: str
    model_prediction: str
    confidence: float
    true_label: str
    features: list[float]

# ✅ /feedback endpoint
@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    if len(feedback.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")

    # Queue the feedback job to Redis
    job = feedback_queue.enqueue(
        save_feedback_job,
        feedback.user_id,
        feedback.model_prediction,
        feedback.confidence,
        feedback.true_label,
        feedback.features
    )

    return {"message": "✅ Feedback submitted", "job_id": job.id}



# Input schema
class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "TensorTrapX (mh) Breast Cancer API is running 🚀"}

@app.post("/predict")
def predict(input: PredictionInput):
    features = input.features
    if len(features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features.")

    X = np.array(features).reshape(1, -1)
    prob = model.predict(X)[0][0]
    label = class_names[int(prob >= 0.5)]

    # 🔁 Metrics update
    prediction_count.inc()
    prediction_confidence.set(prob)
    prediction_label_total.labels(label=label).inc()

    return {
        "prediction": label,
        "probability": round(float(prob), 4)
    }


from lime import lime_tabular
import pandas as pd

# Initialize LIME explainer once with dummy data
feature_names = [f"feature_{i}" for i in range(30)]
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.zeros((1, 30)),  # placeholder; LIME just needs shape
    feature_names=feature_names,
    class_names=["Benign", "Malignant"],
    mode="classification"
)

# Wrapper to match LIME expectations
def predict_proba_wrapper(x):
    preds = model.predict(x)
    return np.hstack([1 - preds, preds])

@app.post("/explain")
def explain(input: PredictionInput):
    features = input.features
    if len(features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features.")

    # Get LIME explanation
    exp = explainer_lime.explain_instance(
        np.array(features),
        predict_proba_wrapper,
        num_features=10
    )

    # Extract feature weights
    explanation = [{"feature": f, "weight": w} for f, w in exp.as_list()]
    return {
        "explanation": explanation,
        "instance": features
    }

from fastapi.responses import Response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
