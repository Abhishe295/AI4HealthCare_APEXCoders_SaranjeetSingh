from fastapi import FastAPI, UploadFile, File
import tempfile, os
from fastapi.middleware.cors import CORSMiddleware

from inference.binary_infer import load_binary_model, predict_binary
from inference.multiclass_infer import load_multiclass_model, predict_multiclass

app = FastAPI(title="NeuroScan AI")

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- LOAD MODELS --------
binary_model = load_binary_model("models/binary_model.pth")
multi_model  = load_multiclass_model("models/multiclass_model.pth")

# -------- STATS --------
STATS = {
    "total": 0,
    "binary": 0,
    "multiclass": 0,
    "history": []
}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------- TEMP SAVE --------
def _save_temp(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file.file.read())
        return tmp.name

# -------- BINARY --------
@app.post("/predict/binary")
async def binary_predict(file: UploadFile = File(...)):
    path = _save_temp(file)

    result = predict_binary(binary_model, path)

    os.remove(path)

    STATS["total"] += 1
    STATS["binary"] += 1
    STATS["history"].append({
        "index": STATS["total"],
        "task": "binary",
        "confidence": result["confidence"]
    })

    return result


# -------- MULTICLASS --------
@app.post("/predict/multiclass")
async def multiclass_predict(file: UploadFile = File(...)):
    path = _save_temp(file)

    result = predict_multiclass(multi_model, path)

    os.remove(path)

    STATS["total"] += 1
    STATS["multiclass"] += 1
    STATS["history"].append({
        "index": STATS["total"],
        "task": "multiclass",
        "probs": result["probabilities"]
    })

    return result


# -------- STATS --------
@app.get("/stats/summary")
def stats_summary():
    return {
        "total_predictions": STATS["total"],
        "binary_predictions": STATS["binary"],
        "multiclass_predictions": STATS["multiclass"]
    }


@app.get("/stats/history")
def stats_history():
    return {"history": STATS["history"]}