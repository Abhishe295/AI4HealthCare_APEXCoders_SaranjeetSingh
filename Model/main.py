from fastapi import FastAPI, UploadFile, File
import numpy as np

from inference.binary_infer import load_binary_model, predict_binary
from inference.multiclass_infer import load_multiclass_model, predict_multiclass

app = FastAPI(title="Alzheimer MRI API")

# ---- load models ONCE ----
binary_model = load_binary_model("models/binary_model.pth")
multi_model  = load_multiclass_model("models/multiclass_model.pth")

@app.get("/")
def health():
    return {"status":"API running"}

@app.post("/predict/binary")
async def binary(file: UploadFile = File(...)):
    volume = np.load(file.file)
    return predict_binary(binary_model, volume)

@app.post("/predict/multiclass")
async def multiclass(file: UploadFile = File(...)):
    volume = np.load(file.file)
    return predict_multiclass(multi_model, volume)
