# Singular Value Decomposition (SVD) for Image Compression Studio

An interactive linear algebra and digital image processing application that demonstrates how **Singular Value Decomposition (SVD)** achieves low-rank image compression, matrix reconstruction, and spectral energy analysis.

---

## 🚀 Features

- **⚡ Interactive Web Application (Streamlit)**: Real-time slider and preset controls for rank truncation ($k$) with instant matrix updates.
- **🎨 Preset Benchmark Gallery & Custom Uploads**: Test instantly with standard benchmarks (*Astronaut, Chelsea Cat, Coffee Cup, Rocket, Cameraman*) or upload your own PNG/JPG/WEBP images.
- **📊 Gold-Standard Quality Metrics**: Real-time evaluation of:
  - **Compression Ratio ($CR$)** & Storage Reduction Percentage
  - **Peak Signal-to-Noise Ratio (PSNR in dB)**
  - **Structural Similarity Index (SSIM)**
  - **Mean Squared Error (MSE)**
  - **Cumulative Frobenius Energy (%)**
- **🔍 Residual Error Heatmap ($|I_{orig} - I_{comp}|$ )**: Inspect pixel-level approximation error patterns to visualize high-frequency edge loss.
- **📈 Interactive Plotly Analytics**: Zoomable, pannable charts for:
  - Singular value decay (scree plot on log scale)
  - Cumulative energy curves with 90%, 95%, 99% threshold guides
  - Rate-Distortion curves (PSNR vs Compression Ratio & MSE vs $k$)
- **🔬 Rank-1 Basis Components Explorer**: Visually inspect individual outer products $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ to understand how spatial frequencies are layered.
- **🎞️ Multi-Rank Visual Comparison Grid**: Side-by-side thumbnail evolution across multiple ranks ($k=2, 10, 30, 80, 150$).
- **📥 Image Export Center**: Save and download compressed images in JPEG or PNG formats.

---

## 🧠 Mathematical Foundation

Any real matrix $A \in \mathbb{R}^{M \times N}$ can be factored via SVD into:
$$A = U \Sigma V^T = \sum_{i=1}^{\min(M,N)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

- $U \in \mathbb{R}^{M \times M}$: Orthogonal matrix of left singular vectors (spatial patterns).
- $\Sigma \in \mathbb{R}^{M \times N}$: Diagonal matrix of sorted singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$.
- $V^T \in \mathbb{R}^{N \times N}$: Orthogonal matrix of right singular vectors.

According to the **Eckart-Young-Mirsky Theorem**, truncating the sum to the top $k$ terms provides the optimal rank-$k$ matrix approximation:
$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

### Storage Reduction
Instead of storing $M \times N$ values per channel, the compressed format stores:
$$\text{Storage} = k \times (M + N + 1) \text{ values per channel}$$

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/svd-image-compression.git
   cd svd-image-compression
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Run

### 1. Modern Web Application (FastAPI + Interactive UI) ⭐ Recommended
```bash
python run.py
```
*Or:*
```bash
python server.py
```
*Launches the ultra-responsive interactive Web Studio at `http://localhost:8000` with instant real-time slider updates, interactive split-wipe curtain slider, dark/light theme toggle, and Chart.js analytics.*

### 2. Streamlit Dashboard (Alternative)
```bash
streamlit run app.py
```
*Launches the Streamlit app at `http://localhost:8501`.*

### 3. Interactive Jupyter Notebook
```bash
jupyter notebook svd_compression.ipynb
```

### 4. Command Line Verification Script
```bash
python test_script.py
```

---

## 📚 Tech Stack & Libraries
- **`FastAPI` & `Uvicorn`** - High-performance backend API serving real-time SVD matrix calculations
- **`Vanilla CSS & HTML5`** - Ultra-modern, responsive custom UI with glassmorphism and theme switching
- **`Chart.js` & `Plotly`** - Interactive analytical charts (decay curves, logarithmic scree plots)
- **`NumPy`** - High-performance linear algebra & `np.linalg.svd`
- **`scikit-image`** - Benchmark image dataset & PSNR metrics
- **`scikit-learn`** - Mean Squared Error (MSE) calculation
- **`Matplotlib`** - Inferno colormap generation for residual error heatmaps
- **`Pillow`** - Image format conversions and exports
- **`Streamlit`** - Python dashboard alternative

