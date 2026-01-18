import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import uvicorn

app = FastAPI(title="fish-classifier")

def preprocess_pytorch_style(X):
    X = X / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std
    
    return X.astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

session = ort.InferenceSession(
    "fish_model.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]


class PredictRequest(BaseModel):
    url: HttpUrl


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

def softmax(x: np.ndarray):
    x = x - np.max(x)          # numerical stability
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def predict(url: str):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})

    logits = result[0][0]              # shape: (num_classes,)
    probs = softmax(logits)            # now in [0, 1], sum to 1

    predictions_dict = {
        cls: float(p)
        for cls, p in zip(classes, probs)
    }

    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_probability = float(probs[top_idx])

    return predictions_dict, top_class, top_probability



@app.get("/")
def root():
    return {"message": "Fish Classification Service"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    predictions, top_class, top_prob = predict(str(request.url))
    
    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)