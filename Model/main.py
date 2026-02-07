from fastapi import FastAPI, UploadFile, File
import numpy as np, tempfile, os
from fastapi.middleware.cors import CORSMiddleware


from inference.binary_infer import load_binary_model, predict_binary
from inference.multiclass_infer import load_multiclass_model, predict_multiclass

app = FastAPI(title="NeuroScan AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://127.0.0.1:5173",
        "http://localhost:3000"    # fallback
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

binary_model = load_binary_model("models/binary_model.pth")
multi_model  = load_multiclass_model("models/multiclass_model.pth")

# -------- In-memory stats (simple & demo-ready) --------
STATS = {
    "total": 0,
    "binary": 0,
    "multiclass": 0,
    "history": []
}

@app.get("/health")
def health():
    return {"status": "ok"}

def _save_temp(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        return tmp.name

@app.post("/predict/binary")
async def binary_predict(file: UploadFile = File(...)):
    path = _save_temp(file)
    volume = np.load(path)
    os.remove(path)

    result = predict_binary(binary_model, volume)

    STATS["total"] += 1
    STATS["binary"] += 1
    STATS["history"].append({
        "index": STATS["total"],
        "task": "binary",
        "confidence": result["confidence"]
    })


    return result

@app.post("/predict/multiclass")
async def multiclass_predict(file: UploadFile = File(...)):
    path = _save_temp(file)
    volume = np.load(path)
    os.remove(path)

    result = predict_multiclass(multi_model, volume)

    STATS["total"] += 1
    STATS["multiclass"] += 1
    STATS["history"].append({
        "index": STATS["total"],
        "task": "multiclass",
        "probs": result["probabilities"]
    })


    return result

@app.get("/stats/summary")
def stats_summary():
    return {
        "total_predictions": STATS["total"],
        "binary_predictions": STATS["binary"],
        "multiclass_predictions": STATS["multiclass"]
    }

@app.get("/stats/history")
def stats_history():
    return { "history": STATS["history"] }
