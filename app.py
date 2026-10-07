import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.cm as cm
import plotly.graph_objects as go
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SVD Image Compression Demo",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Clean & Appealing Theme CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }
    
    html, body, [class*="css"], .stApp {
        font-family: var(--font-sans);
    }
    
    /* Global Page Clean Spacing */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1280px;
    }
    
    /* Top Presentation Header */
    .pres-header {
        padding: 1.3rem 1.6rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        position: relative;
        overflow: hidden;
    }
    
    .pres-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
    }
    
    .pres-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .pres-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        font-weight: 500;
        margin: 0.25rem 0 0 0;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.05rem 1.2rem;
        text-align: left;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03), 0 4px 10px rgba(15, 23, 42, 0.02);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        min-height: 112px;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
        border-color: #cbd5e1;
    }
    
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    
    .kpi-badge {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 20px;
        letter-spacing: 0.01em;
    }
    
    /* Badge Color Palette */
    .badge-cyan, .badge-blue { 
        background: #eff6ff; 
        color: #2563eb; 
        border: 1px solid #bfdbfe; 
    }
    .badge-emerald { 
        background: #ecfdf5; 
        color: #059669; 
        border: 1px solid #a7f3d0; 
    }
    .badge-purple { 
        background: #f5f3ff; 
        color: #7c3aed; 
        border: 1px solid #ddd6fe; 
    }
    .badge-amber { 
        background: #fffbeb; 
        color: #d97706; 
        border: 1px solid #fde68a; 
    }
    .badge-rose { 
        background: #fff1f2; 
        color: #e11d48; 
        border: 1px solid #fecdd3; 
    }
    .badge-slate {
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-top: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.65;
    }
    
    .sidebar-stat-card code {
        background: #f1f5f9;
        color: #0f172a;
        padding: 0.12rem 0.4rem;
        border-radius: 5px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 600;
    }
    
    /* Button Aesthetics */
    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        font-size: 0.84rem;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        color: #334155;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: all 0.15s ease-in-out;
        padding: 0.45rem 0.75rem;
    }
    
    div.stButton > button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
        background-color: #f5f3ff;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.1);
    }
    
    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* Download Button */
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.88rem;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: #ffffff;
        border: none;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
        padding: 0.6rem 1rem;
    }
    
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        transform: translateY(-1px);
        color: #ffffff;
    }

    /* Image Display Styling */
    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
    
    /* Section Dividers */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #f1f5f9 !important;
    }
    
    /* Alert / Takeaway styling */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #e0e7ff;
        background-color: #f8fafc;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    /* Dark Mode Adaptive Support */
    @media (prefers-color-scheme: dark) {
        .pres-header, .kpi-card, .sidebar-stat-card {
            background: #1e293b !important;
            border-color: #334155 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }
        .pres-title {
            color: #f8fafc !important;
        }
        .pres-subtitle, .kpi-label {
            color: #94a3b8 !important;
        }
        .sidebar-stat-card {
            color: #cbd5e1 !important;
        }
        .sidebar-stat-card code {
            background: #0f172a !important;
            color: #38bdf8 !important;
        }
        div.stButton > button {
            background-color: #1e293b !important;
            border-color: #334155 !important;
            color: #f1f5f9 !important;
        }
        div.stButton > button:hover {
            background-color: #2e3856 !important;
            border-color: #6366f1 !important;
            color: #818cf8 !important;
        }
        div[data-testid="stImage"] {
            background: #1e293b !important;
            border-color: #334155 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right-color: #334155 !important;
        }
        div[data-testid="stAlert"] {
            background-color: #1e293b !important;
            border-color: #334155 !important;
        }
        .badge-cyan, .badge-blue { background: rgba(59, 130, 246, 0.15) !important; color: #60a5fa !important; border-color: rgba(59, 130, 246, 0.3) !important; }
        .badge-emerald { background: rgba(16, 185, 129, 0.15) !important; color: #34d399 !important; border-color: rgba(16, 185, 129, 0.3) !important; }
        .badge-purple { background: rgba(139, 92, 246, 0.15) !important; color: #a78bfa !important; border-color: rgba(139, 92, 246, 0.3) !important; }
        .badge-amber { background: rgba(245, 158, 11, 0.15) !important; color: #fbbf24 !important; border-color: rgba(245, 158, 11, 0.3) !important; }
        .badge-rose { background: rgba(244, 63, 94, 0.15) !important; color: #fb7185 !important; border-color: rgba(244, 63, 94, 0.3) !important; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Core Helper Functions & Cached Matrix Operations
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_benchmark_image(name):
    """Loads benchmark image from standard skimage library."""
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


@st.cache_data(show_spinner=False)
def compute_svd(img_norm):
    """
    Computes SVD per RGB channel once and caches it in memory.
    """
    Ur, Sr, VTr = np.linalg.svd(img_norm[:, :, 0], full_matrices=False)
    Ug, Sg, VTg = np.linalg.svd(img_norm[:, :, 1], full_matrices=False)
    Ub, Sb, VTb = np.linalg.svd(img_norm[:, :, 2], full_matrices=False)
    return (Ur, Sr, VTr), (Ug, Sg, VTg), (Ub, Sb, VTb)


def reconstruct_image(k, svd_r, svd_g, svd_b):
    """
    Reconstructs image from top k singular values.
    """
    Ur, Sr, VTr = svd_r
    Ug, Sg, VTg = svd_g
    Ub, Sb, VTb = svd_b

    r = Ur[:, :k] @ np.diag(Sr[:k]) @ VTr[:k, :]
    g = Ug[:, :k] @ np.diag(Sg[:k]) @ VTg[:k, :]
    b = Ub[:, :k] @ np.diag(Sb[:k]) @ VTb[:k, :]

    recon = np.stack([r, g, b], axis=2)
    return np.clip(recon, 0.0, 1.0)


def generate_residual_heatmap(orig, compressed):
    """
    Computes absolute error |orig - compressed| and maps it to an inferno colormap.
    Returns: (heatmap_rgb, max_error, mean_error)
    """
    diff = np.abs(orig - compressed)
    # Average across RGB channels to get per-pixel error magnitude
    diff_mag = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
    max_err = float(np.max(diff_mag))
    mean_err = float(np.mean(diff_mag))

    # Normalize error to [0, 1] with dynamic scaling for clear visualization
    if max_err > 1e-6:
        diff_norm = np.clip(diff_mag / max(max_err, 0.05), 0.0, 1.0)
    else:
        diff_norm = np.zeros_like(diff_mag)

    # Apply Inferno colormap (Dark purple = 0 error, Bright Yellow/White = Max error)
    heatmap_rgba = cm.inferno(diff_norm)
    heatmap_rgb = np.clip(heatmap_rgba[:, :, :3], 0.0, 1.0)
    return heatmap_rgb, max_err, mean_err


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Demo Setup")
    
    img_choice = st.selectbox(
        "Image Source",
        ["Astronaut", "Chelsea (Cat)", "Rocket", "Coffee Cup", "Cameraman", "Upload Custom Image..."]
    )
    
    if img_choice == "Upload Custom Image...":
        uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            raw_img = np.array(pil_img)
        else:
            raw_img = load_benchmark_image("Astronaut")
    else:
        raw_img = load_benchmark_image(img_choice)

    # Normalize image
    img_norm = raw_img.astype(float) / 255.0
    if len(img_norm.shape) == 2:
        img_norm = np.stack([img_norm, img_norm, img_norm], axis=2)
    elif img_norm.shape[2] > 3:
        img_norm = img_norm[:, :, :3]

    H, W, _ = img_norm.shape
    max_k = min(H, W)

    # Pre-compute SVD
    with st.spinner("Computing SVD matrices..."):
        svd_r, svd_g, svd_b = compute_svd(img_norm)

    st.markdown("---")
    st.markdown("### ⚡ Live Rank Control ($k$)")

    # State management for slider
    if "k_val" not in st.session_state:
        st.session_state.k_val = min(30, max_k)
    st.session_state.k_val = int(min(max(1, st.session_state.k_val), max_k))

    def set_k(target):
        st.session_state.k_val = int(min(max(1, target), max_k))

    # Fast Presets
    st.caption("⚡ 1-Click Presentation Presets:")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.button("🏎️ k = 5 (Low)", use_container_width=True, on_click=set_k, args=(min(5, max_k),))
        st.button("⚖️ k = 30 (Mid)", use_container_width=True, on_click=set_k, args=(min(30, max_k),))
    with col_b2:
        st.button("💎 k = 80 (High)", use_container_width=True, on_click=set_k, args=(min(80, max_k),))
        st.button("🌟 k = 150 (Max)", use_container_width=True, on_click=set_k, args=(min(150, max_k),))

    # Live Slider
    k = st.slider(
        f"Singular Values Retained (1 to {max_k})",
        min_value=1,
        max_value=max_k,
        value=st.session_state.k_val,
        key="slider_widget",
        on_change=lambda: setattr(st.session_state, "k_val", st.session_state.slider_widget)
    )
    st.session_state.k_val = k

    st.markdown("---")
    view_mode = st.radio(
        "Display Mode",
        ["🖼️ Side-by-Side (Original vs SVD)", "🔍 3-Way Inspection (Original + SVD + Error Map)"],
        index=0
    )
    
    st.markdown(f"""
    <div class="sidebar-stat-card">
        <b>Dimensions:</b> <code>{W} × {H}</code> px<br>
        <b>Max Matrix Rank:</b> <code>{max_k}</code><br>
        <b>Active Rank ($k$):</b> <code>{k}</code> ({k/max_k*100:.1f}%)
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Dynamic Calculations for Active k
# ---------------------------------------------------------
compressed_img = reconstruct_image(k, svd_r, svd_g, svd_b)
heatmap_img, max_error_val, mean_error_val = generate_residual_heatmap(img_norm, compressed_img)

# Memory & Compression Ratio
raw_bytes = H * W * 3 * 8
comp_bytes = ((H * k) + k + (k * W)) * 3 * 8
compression_ratio = raw_bytes / comp_bytes
space_saved = (1.0 - (comp_bytes / raw_bytes)) * 100.0

# Mathematical Metrics
mse_val = mean_squared_error(img_norm.reshape(-1), compressed_img.reshape(-1))
psnr_val = psnr(img_norm, compressed_img, data_range=1.0)

# Frobenius Energy %
energy_r = (np.sum(svd_r[1][:k]**2) / np.sum(svd_r[1]**2)) * 100.0
energy_g = (np.sum(svd_g[1][:k]**2) / np.sum(svd_g[1]**2)) * 100.0
energy_b = (np.sum(svd_b[1][:k]**2) / np.sum(svd_b[1]**2)) * 100.0
avg_energy = (energy_r + energy_g + energy_b) / 3.0


# ---------------------------------------------------------
# Header & Top Action Bar
# ---------------------------------------------------------
st.markdown(f"""
<div class="pres-header">
    <div>
        <h1 class="pres-title">✨ SVD Image Compression Studio</h1>
        <p class="pres-subtitle">Interactive Matrix Factorization • Rank Truncation Demo • $A_k = U_k \\Sigma_k V_k^T$</p>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="kpi-badge badge-blue" style="font-size: 0.82rem; padding: 0.35rem 0.75rem;">Active Rank: k = {k}</span>
        <span class="kpi-badge badge-emerald" style="font-size: 0.82rem; padding: 0.35rem 0.75rem;">{space_saved:.1f}% Space Saved</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2 Key Metric Cards in 1 Row
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Compression Ratio</div>
        <div class="kpi-value" style="color: #2563eb;">{compression_ratio:.1f}x</div>
        <span class="kpi-badge badge-blue">{space_saved:.1f}% Reduction</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Energy Retained</div>
        <div class="kpi-value" style="color: #059669;">{avg_energy:.2f}%</div>
        <span class="kpi-badge badge-emerald">Frobenius Norm</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Centerpiece: Live Visual Comparison & Residual Error Map
# ---------------------------------------------------------
st.markdown("---")

if view_mode == "🔍 3-Way Inspection (Original + SVD + Error Map)":
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown(f"### 🖼️ Original (`{W} × {H}`)")
        st.image(img_norm, use_container_width=True)
        st.caption(f"💾 Size: **{raw_bytes / 1024:.1f} KB**")

    with v2:
        st.markdown(f"### ⚡ SVD Compressed ($k = {k}$)")
        st.image(compressed_img, use_container_width=True)
        st.caption(f"💾 Size: **{comp_bytes / 1024:.1f} KB** | Ratio: **{compression_ratio:.1f}x**")

    with v3:
        st.markdown(f"### 🔥 Residual Error Map")
        st.image(heatmap_img, use_container_width=True)
        st.caption(f"🔴 Bright = High Error | Peak: **{max_error_val:.3f}** | Avg: **{mean_error_val:.4f}**")

else:  # Side-by-Side default
    v1, v2 = st.columns(2)
    with v1:
        st.markdown(f"### 🖼️ Original Uncompressed (`{W} × {H}`)")
        st.image(img_norm, use_container_width=True)
        st.caption(f"💾 Storage Size: **{raw_bytes / 1024:.1f} KB** (Float64 Matrix)")

    with v2:
        st.markdown(f"### ⚡ SVD Reconstruction ($k = {k}$)")
        st.image(compressed_img, use_container_width=True)
        st.caption(f"💾 Factorized Size: **{comp_bytes / 1024:.1f} KB** | Ratio: **{compression_ratio:.1f}x**")


# ---------------------------------------------------------
# Interactive Singular Value Spectrum (Plotly)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Singular Value Decay Curve (The Mathematical Explanation)")
st.markdown("Notice how singular values $\\sigma_i$ plunge exponentially after the first few ranks — explaining why low-rank approximations capture nearly all image structure.")

fig = go.Figure()
indices = np.arange(1, len(svd_r[1]) + 1)

# R, G, B Curves
fig.add_trace(go.Scatter(
    x=indices, y=svd_r[1], mode='lines', name='Red Channel σ',
    line=dict(color='#ef4444', width=2),
    hovertemplate="Rank %{x}: σ = %{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=indices, y=svd_g[1], mode='lines', name='Green Channel σ',
    line=dict(color='#10b981', width=2),
    hovertemplate="Rank %{x}: σ = %{y:.2f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=indices, y=svd_b[1], mode='lines', name='Blue Channel σ',
    line=dict(color='#3b82f6', width=2),
    hovertemplate="Rank %{x}: σ = %{y:.2f}<extra></extra>"
))

# Dynamic vertical indicator at active k
fig.add_vline(
    x=k, line_dash="dash", line_color="#4f46e5", line_width=2,
    annotation_text=f"Active Rank k = {k} ({avg_energy:.1f}% Energy)",
    annotation_position="top right",
    annotation_font_color="#4f46e5",
    annotation_font_size=11,
    annotation_bgcolor="rgba(238, 242, 255, 0.9)",
    annotation_bordercolor="#c7d2fe",
    annotation_borderwidth=1,
    annotation_borderpad=4
)

fig.update_layout(
    xaxis_title="Singular Value Rank Index (i)",
    yaxis_title="Singular Value Magnitude (σ_i, Log Scale)",
    yaxis_type="log",
    template="plotly_white",
    height=330,
    paper_bgcolor="rgba(255, 255, 255, 0)",
    plot_bgcolor="rgba(255, 255, 255, 0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#475569"),
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    xaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        zeroline=False
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        zeroline=False
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Compact Mathematical Takeaway for Presentation & Download
# ---------------------------------------------------------
bot_col1, bot_col2 = st.columns([3, 1])

with bot_col1:
    st.info(f"""
    **📐 Presentation Formula Summary:**  
    Instead of storing **{H * W:,}** numbers per channel, SVD stores only **$k(M + N + 1) = {k} \\times ({H} + {W} + 1) = {((H * k) + k + (k * W)):,}$** numbers per channel.  
    With **$k = {k}$**, this delivers **{compression_ratio:.1f}x compression** while preserving **{avg_energy:.2f}%** of the total image Frobenius energy.
    """)

with bot_col2:
    comp_uint8 = (compressed_img * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(comp_uint8).save(buf, format="JPEG", quality=95)
    st.download_button(
        label=f"📥 Download (k={k})",
        data=buf.getvalue(),
        file_name=f"svd_k{k}.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
