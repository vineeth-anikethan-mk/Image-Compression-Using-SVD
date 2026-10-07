# ⚡ SVD Image Compression & Matrix Spectral Lab

An interactive, high-performance linear algebra and digital image processing laboratory demonstrating how **Singular Value Decomposition (SVD)** achieves optimal low-rank matrix approximations and digital image compression via rank truncation ($A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$).

---

## 🚀 Key Features

- **⚡ Real-Time Rank Truncation Scrubber**: Sub-millisecond matrix reconstruction powered by NumPy SVD vectorization and FastAPI, with direct numeric input and preset chips ($k = 5, 30, 80, 150$).
- **🪟 Interactive Split-Wipe Curtain**: Drag-and-wipe visual comparison between uncompressed full-rank float matrices and truncated rank-$k$ approximations in a unified viewport.
- **🖼️ Flexible Visual Workspaces**:
  - **Split Wipe**: Draggable hairline divider comparing original reference and reconstructed matrix.
  - **Dual View**: Side-by-side comparison with storage size metrics.
  - **3-Way Inspect**: Reference image, reconstructed matrix, and channel comparison.
- **📈 Logarithmic Singular Value Spectrum**: Real-time Chart.js scree plot demonstrating exponential decay across RGB channels ($\sigma_i$) with a dynamic active rank crosshair.
- **📊 Compression & Energy Telemetry HUD**: Live tracking of:
  - **Compression Ratio ($CR$)** & Storage Reduction Percentage
  - **Cumulative Frobenius Energy Retention** ($\|\cdot\|_F$) with a dynamic visual meter
  - **Per-Channel Matrix Storage Math**: Exact count of numbers stored per channel ($k(M + N + 1)$ vs. $M \times N$).
- **🎨 Preset Benchmark Gallery & Custom Import**: Instant testing on standard scientific datasets (*Astronaut, Chelsea Cat, Rocket, Coffee Cup, Cameraman*) or custom drag-and-drop image uploads.
- **🌓 Studio Theme Toggle**: Seamless switching between Obsidian Dark and Architectural Light modes.
- **📥 1-Click High-Fidelity Export**: Download the reconstructed image as a JPEG at the active rank $k$.
- **⌨️ Power-User Keyboard Shortcuts**:
  - `←` / `→` (or `[` / `]`): Increment or decrement rank $k$ by 1 (or by 10 with `Shift`).
  - `1`, `2`, `3`: Switch view modes (*Split Wipe, Dual View, 3-Way Inspect*).
  - `T`: Toggle between Dark and Light themes.

---

## 📐 Mathematical Foundation

Every real channel matrix $A \in \mathbb{R}^{M \times N}$ can be factored via SVD into two orthogonal matrices and a diagonal matrix of singular values:

$$A = U \Sigma V^T = \sum_{i=1}^{\min(M, N)} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

Where:
- $U \in \mathbb{R}^{M \times M}$: Orthogonal matrix of left singular vectors (spatial basis).
- $\Sigma \in \mathbb{R}^{M \times N}$: Diagonal matrix of singular values in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$).
- $V^T \in \mathbb{R}^{N \times N}$: Orthogonal matrix of right singular vectors.

### Optimal Low-Rank Approximation (Eckart–Young–Mirsky Theorem)
Truncating the summation to the top $k$ singular values gives the mathematically optimal rank-$k$ approximation minimizing Frobenius norm error:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

$$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{\min(M, N)} \sigma_i^2}$$

### Frobenius Energy Retention
$$\text{Energy}(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{\min(M, N)} \sigma_i^2} \times 100\%$$

### Storage Complexity & Compression
Instead of storing $M \times N$ values per channel, storing the factorized rank-$k$ components requires:

$$\text{Values Stored per Channel} = k \times (M + N + 1)$$

$$\text{Compression Ratio (CR)} = \frac{M \times N}{k(M + N + 1)}$$

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/svd-image-compression.git
   cd "Image Compression using SVD"
   ```

2. **Create and activate a virtual environment (optional but recommended):**
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
*Or directly via server:*
```bash
python server.py
```
*Launches the studio at `http://localhost:8000` with the interactive split-wipe canvas, real-time rank scrubber, Chart.js scree plot, and theme toggle.*

### 2. Streamlit Dashboard (Alternative)
```bash
streamlit run app.py
```
*Launches the Streamlit app at `http://localhost:8501`.*

### 3. Interactive Jupyter Notebook
```bash
jupyter notebook svd_compression.ipynb
```

### 4. Verification Test Script
```bash
python test_script.py
```

---

## 📚 Tech Stack

- **Backend**: `FastAPI`, `Uvicorn`, `Python 3.10+`
- **Frontend**: `Vanilla JavaScript (ES6+)`, `HTML5`, `Custom CSS Design System`
- **Visual Analytics**: `Chart.js` (Singular Value Spectrum Decay Scree Plot)
- **Scientific Computing**: `NumPy` (`np.linalg.svd`), `scikit-image`
- **Image Processing**: `Pillow (PIL)`, `Matplotlib`
- **Alternative Frameworks**: `Streamlit`, `Plotly`
