import io
import base64
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

app = FastAPI(title="SVD Image Compression Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state for active image & SVD decomposition
state = {
    "name": "Astronaut",
    "img_norm": None,
    "H": 0,
    "W": 0,
    "max_k": 0,
    "svd_r": None,
    "svd_g": None,
    "svd_b": None,
    "energy_total_r": 0.0,
    "energy_total_g": 0.0,
    "energy_total_b": 0.0,
    "orig_b64": "",
    "raw_bytes": 0
}

def load_benchmark_array(name: str):
    if name == "Astronaut":
        return data.astronaut()
    elif name == "Chelsea (Cat)":
        return data.chelsea()
    elif name == "Rocket":
        return data.rocket()
    elif name == "Coffee Cup":
        return data.coffee()
    elif name == "Cameraman":
        cam = data.camera()
        return np.stack([cam, cam, cam], axis=2)
    return data.astronaut()

def array_to_b64_jpeg(arr: np.ndarray, quality: int = 95) -> str:
    uint8_img = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def process_and_cache_image(raw_img: np.ndarray, name: str = "Image"):
    img_norm = raw_img.astype(float) / 255.0
    if len(img_norm.shape) == 2:
        img_norm = np.stack([img_norm, img_norm, img_norm], axis=2)
    elif img_norm.shape[2] > 3:
        img_norm = img_norm[:, :, :3]

    H, W, _ = img_norm.shape
    max_k = min(H, W)

    Ur, Sr, VTr = np.linalg.svd(img_norm[:, :, 0], full_matrices=False)
    Ug, Sg, VTg = np.linalg.svd(img_norm[:, :, 1], full_matrices=False)
    Ub, Sb, VTb = np.linalg.svd(img_norm[:, :, 2], full_matrices=False)

    state["name"] = name
    state["img_norm"] = img_norm
    state["H"] = H
    state["W"] = W
    state["max_k"] = max_k
    state["svd_r"] = (Ur, Sr, VTr)
    state["svd_g"] = (Ug, Sg, VTg)
    state["svd_b"] = (Ub, Sb, VTb)
    state["energy_total_r"] = float(np.sum(Sr**2))
    state["energy_total_g"] = float(np.sum(Sg**2))
    state["energy_total_b"] = float(np.sum(Sb**2))
    state["raw_bytes"] = H * W * 3 * 8
    state["orig_b64"] = array_to_b64_jpeg(img_norm)

    # Prepare downsampled spectrum curve data for chart (with positive minimum for log scale)
    step = max(1, max_k // 150)
    indices = list(range(1, max_k + 1, step))
    if indices[-1] != max_k:
        indices.append(max_k)

    def clean_sigma(arr, idx):
        val = float(arr[idx - 1])
        return max(val, 0.001)

    chart_data = {
        "indices": indices,
        "sigma_r": [clean_sigma(Sr, i) for i in indices],
        "sigma_g": [clean_sigma(Sg, i) for i in indices],
        "sigma_b": [clean_sigma(Sb, i) for i in indices],
    }
    state["chart_data"] = chart_data

    return {
        "name": name,
        "width": W,
        "height": H,
        "max_k": max_k,
        "raw_kb": round(state["raw_bytes"] / 1024, 1),
        "orig_b64": state["orig_b64"],
        "chart_data": chart_data
    }

# Initialize with default Astronaut image
process_and_cache_image(load_benchmark_array("Astronaut"), "Astronaut")

class SetImageRequest(BaseModel):
    source: str

class CompressRequest(BaseModel):
    k: int

@app.get("/api/images")
def get_benchmark_images():
    return {
        "presets": ["Astronaut", "Chelsea (Cat)", "Rocket", "Coffee Cup", "Cameraman"],
        "active": state["name"],
        "width": state["W"],
        "height": state["H"],
        "max_k": state["max_k"],
        "raw_kb": round(state["raw_bytes"] / 1024, 1),
        "orig_b64": state["orig_b64"],
        "chart_data": state.get("chart_data")
    }

@app.post("/api/set-image")
def set_preset_image(req: SetImageRequest):
    if req.source not in ["Astronaut", "Chelsea (Cat)", "Rocket", "Coffee Cup", "Cameraman"]:
        raise HTTPException(status_code=400, detail="Invalid benchmark image name")
    arr = load_benchmark_array(req.source)
    data_res = process_and_cache_image(arr, req.source)
    return data_res

@app.post("/api/upload")
async def upload_custom_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_img = np.array(pil_img)
        # Cap large dimensions to prevent memory overflow (e.g. max 1024x1024)
        if raw_img.shape[0] > 1024 or raw_img.shape[1] > 1024:
            pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            raw_img = np.array(pil_img)
        data_res = process_and_cache_image(raw_img, file.filename or "Custom Upload")
        return data_res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded image: {str(e)}")

@app.post("/api/compress")
def compress_image(req: CompressRequest):
    if state["img_norm"] is None:
        raise HTTPException(status_code=400, detail="No image loaded")

    k = max(1, min(req.k, state["max_k"]))
    H, W = state["H"], state["W"]
    Ur, Sr, VTr = state["svd_r"]
    Ug, Sg, VTg = state["svd_g"]
    Ub, Sb, VTb = state["svd_b"]

    # Reconstruct RGB channels
    r = Ur[:, :k] @ np.diag(Sr[:k]) @ VTr[:k, :]
    g = Ug[:, :k] @ np.diag(Sg[:k]) @ VTg[:k, :]
    b = Ub[:, :k] @ np.diag(Sb[:k]) @ VTb[:k, :]

    compressed_img = np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0)
    orig_img = state["img_norm"]

    # Metrics
    raw_bytes = state["raw_bytes"]
    comp_bytes = ((H * k) + k + (k * W)) * 3 * 8
    compression_ratio = raw_bytes / comp_bytes
    space_saved = (1.0 - (comp_bytes / raw_bytes)) * 100.0

    mse_val = float(mean_squared_error(orig_img.reshape(-1), compressed_img.reshape(-1)))
    psnr_val = float(psnr(orig_img, compressed_img, data_range=1.0))

    # Energy %
    energy_r = (float(np.sum(Sr[:k]**2)) / state["energy_total_r"]) * 100.0
    energy_g = (float(np.sum(Sg[:k]**2)) / state["energy_total_g"]) * 100.0
    energy_b = (float(np.sum(Sb[:k]**2)) / state["energy_total_b"]) * 100.0
    avg_energy = (energy_r + energy_g + energy_b) / 3.0

    # Residual Heatmap
    diff = np.abs(orig_img - compressed_img)
    diff_mag = np.mean(diff, axis=2)
    max_err = float(np.max(diff_mag))
    mean_err = float(np.mean(diff_mag))

    if max_err > 1e-6:
        diff_norm = np.clip(diff_mag / max(max_err, 0.05), 0.0, 1.0)
    else:
        diff_norm = np.zeros_like(diff_mag)

    heatmap_rgba = cm.inferno(diff_norm)
    heatmap_rgb = np.clip(heatmap_rgba[:, :, :3], 0.0, 1.0)

    # Base64 encodings
    comp_b64 = array_to_b64_jpeg(compressed_img, quality=95)
    heatmap_b64 = array_to_b64_jpeg(heatmap_rgb, quality=95)

    return {
        "k": k,
        "max_k": state["max_k"],
        "width": W,
        "height": H,
        "comp_kb": round(comp_bytes / 1024, 1),
        "raw_kb": round(raw_bytes / 1024, 1),
        "compression_ratio": round(compression_ratio, 1),
        "space_saved_pct": round(space_saved, 1),
        "energy_pct": round(avg_energy, 2),
        "psnr_db": round(psnr_val, 1),
        "mse": round(mse_val, 5),
        "max_error": round(max_err, 4),
        "mean_error": round(mean_err, 4),
        "comp_b64": comp_b64,
        "heatmap_b64": heatmap_b64,
        "formula": {
            "uncompressed_values": H * W,
            "compressed_values": (H * k) + k + (k * W),
            "k": k,
            "H": H,
            "W": W
        }
    }

@app.get("/api/download")
def download_compressed_image(k: int = 30):
    if state["img_norm"] is None:
        raise HTTPException(status_code=400, detail="No image loaded")

    k = max(1, min(k, state["max_k"]))
    Ur, Sr, VTr = state["svd_r"]
    Ug, Sg, VTg = state["svd_g"]
    Ub, Sb, VTb = state["svd_b"]

    r = Ur[:, :k] @ np.diag(Sr[:k]) @ VTr[:k, :]
    g = Ug[:, :k] @ np.diag(Sg[:k]) @ VTg[:k, :]
    b = Ub[:, :k] @ np.diag(Sb[:k]) @ VTb[:k, :]

    recon = np.clip(np.stack([r, g, b], axis=2), 0.0, 1.0)
    uint8_img = (recon * 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    filename = f"svd_{state['name'].lower().replace(' ', '_')}_k{k}.jpg"
    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# Static files mount
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import webbrowser
    port = 8000
    print(f"🚀 Starting SVD Image Compression Studio at http://localhost:{port}")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
