import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from tensorflow.keras.models import load_model # type: ignore
import joblib

# Load the trained model and scaler (wrapped in helper functions to reload dynamically)
def load_trained_model():
    import os
    if os.path.exists('nbaiot_autoencoder.h5'):
        try:
            return load_model('nbaiot_autoencoder.h5', compile=False)
        except Exception:
            pass
    return None

def load_trained_scaler():
    import os
    if os.path.exists('scaler.save'):
        try:
            return joblib.load('scaler.save')
        except Exception:
            pass
    return None


# ============================================================
# PAGE CONFIGURATION & REDESIGNED THEME
# ============================================================
st.set_page_config(
    page_title="N-BaloT Autoencoder: Adaptive Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Redesigned Premium Dark Cyber Styling
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== GLOBAL APP CONFIG ===== */
.stApp {
    background: #060814 !important;
    font-family: 'Inter', sans-serif !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] {
    background-color: #0b0e1e !important;
    border-right: 1px solid rgba(49, 59, 124, 0.3) !important;
    z-index: 100 !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0 !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ===== HEADER REDESIGN ===== */
.main-header {
    background: linear-gradient(135deg, rgba(13, 18, 43, 0.95) 0%, rgba(20, 24, 53, 0.95) 100%) !important;
    padding: 1.25rem 2rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    box-shadow: 0 8px 32px rgba(8, 12, 36, 0.8) !important;
    margin-bottom: 2rem !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    position: relative !important;
    overflow: hidden !important;
}
.header-left {
    display: flex !important;
    align-items: center !important;
    gap: 20px !important;
}
.header-icon-container {
    width: 60px !important;
    height: 60px !important;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.25) 0%, rgba(14, 165, 233, 0.05) 100%) !important;
    border: 1px solid rgba(14, 165, 233, 0.4) !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #38bdf8 !important;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.25) !important;
}
.header-titles h2 {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 !important;
}
.header-titles h2 span.highlight {
    color: #38bdf8 !important;
}
.header-titles p {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    margin: 4px 0 0 0 !important;
    font-weight: 400 !important;
}
.header-right {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 6px !important;
}
.live-clock-badge {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    background: rgba(13, 18, 43, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
}
.live-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: rgba(16, 185, 129, 0.15) !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: #10b981 !important;
}
.live-dot {
    width: 6px !important;
    height: 6px !important;
    background: #10b981 !important;
    border-radius: 50% !important;
    animation: pulse-live 1.5s infinite !important;
}
@keyframes pulse-live {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    50% { opacity: 0.4; box-shadow: 0 0 0 5px rgba(16,185,129,0); }
}
.time-display {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: #ef4444 !important;
    font-family: 'Inter', monospace !important;
    letter-spacing: 0.5px !important;
}
.date-display {
    font-size: 0.75rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
}

/* ===== CUSTOM METRIC CARDS REDESIGN ===== */
.metric-card-custom {
    background: #0d122b !important;
    border: 1px solid rgba(49, 59, 124, 0.4) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    min-height: 120px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}
.metric-card-custom:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    transform: translateY(-2px) !important;
}
.metric-top {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
}
.metric-icon-container {
    width: 48px !important;
    height: 48px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.metric-icon-container.blue {
    background: rgba(14, 165, 233, 0.1) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.2) !important;
}
.metric-icon-container.red {
    background: rgba(239, 68, 68, 0.1) !important;
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
}
.metric-icon-container.purple {
    background: rgba(168, 85, 247, 0.1) !important;
    color: #c084fc !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
}
.metric-icon-container.orange {
    background: rgba(249, 115, 22, 0.1) !important;
    color: #fb923c !important;
    border: 1px solid rgba(249, 115, 22, 0.2) !important;
}
.metric-text-container {
    display: flex !important;
    flex-direction: column !important;
}
.metric-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
}
.metric-value.red-text { color: #f87171 !important; }
.metric-value.orange-text { color: #fb923c !important; }
.metric-label {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    margin-top: 4px !important;
    font-family: 'Inter', sans-serif !important;
}
.metric-sparkline {
    margin-top: 12px !important;
    height: 30px !important;
    width: 100% !important;
}

/* ===== CHART CARDS REDESIGN ===== */
.chart-card {
    background: #0d122b !important;
    border: 1px solid rgba(49, 59, 124, 0.4) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}
.chart-header-title {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin-bottom: 0.5rem !important;
}

/* ===== BOTTOM CARDS & GAUGE OVERLAYS ===== */
.gauge-card-relative {
    background: #0d122b !important;
    border: 1px solid rgba(49, 59, 124, 0.4) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 0.5rem !important;
}
.chart-title {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
.donut-legend {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    height: 140px !important;
    gap: 12px !important;
    padding-top: 15px !important;
}
.legend-item {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.legend-item .dot {
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    display: inline-block !important;
}
.legend-item .dot.red { background-color: #ef4444 !important; }
.legend-item .dot.orange { background-color: #f97316; }
.legend-item .dot.yellow { background-color: #fbbf24 !important; }
.legend-item .lbl {
    font-weight: 400 !important;
}
.legend-item .val {
    margin-left: auto !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
.legend-item .val .pct {
    color: #64748b !important;
    font-weight: 400 !important;
    font-size: 0.75rem !important;
}
.gauge-overlay-absolute {
    position: absolute !important;
    bottom: 38px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    text-align: center !important;
    background: transparent !important;
    z-index: 5 !important;
    pointer-events: none !important;
}
.gauge-status-text {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.gauge-status-text.green { color: #10b981 !important; }
.gauge-status-text.orange { color: #f97316; }
.gauge-status-text.red { color: #ef4444 !important; }
.gauge-status-sub {
    font-size: 0.75rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    margin-top: 1px !important;
}
.gauge-status-icon {
    margin-top: 6px !important;
}
.gauge-status-icon.green { color: #10b981 !important; }
.gauge-status-icon.orange { color: #f97316 !important; }
.gauge-status-icon.red { color: #ef4444 !important; }

/* ===== FOOTER REDESIGN ===== */
.footer-bar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background: #0d122b !important;
    border: 1px solid rgba(49, 59, 124, 0.4) !important;
    border-radius: 12px !important;
    padding: 1rem 2.5rem !important;
    margin-top: 0.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}
.footer-item {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}
.footer-icon-div {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2) !important;
}
.footer-icon-div.green {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
}
.footer-icon-div.blue {
    background: rgba(14, 165, 233, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.3) !important;
}
.footer-icon-div.purple {
    background: rgba(168, 85, 247, 0.15) !important;
    color: #c084fc !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
}
.footer-text-group {
    display: flex !important;
    flex-direction: column !important;
}
.footer-label {
    font-size: 0.7rem !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
}
.footer-val {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    margin-top: 2px !important;
}
.footer-val.green { color: #10b981 !important; }
.footer-val.blue { color: #38bdf8 !important; }
.footer-val.purple { color: #c084fc !important; }

/* ===== SIDEBAR WIDGET REDESIGNS ===== */
/* Sliders styling */
div[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 8px !important;
}
div[data-testid="stSidebar"] div[data-baseweb="slider"] > div {
    background: #10152e !important;
    border-radius: 4px !important;
    height: 6px !important;
}
div[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%) !important;
}
div[data-testid="stSidebar"] div[data-baseweb="slider"] > div > div > div {
    background-color: #ffffff !important;
    border: 2px solid #8b5cf6 !important;
    box-shadow: 0 0 8px rgba(139, 92, 246, 0.6) !important;
    width: 16px !important;
    height: 16px !important;
}
/* Info card */
.sidebar-info-card {
    background: rgba(13, 18, 43, 0.4) !important;
    border: 1px solid rgba(49, 59, 124, 0.4) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    margin-top: 1rem !important;
}
.sidebar-info-card .info-title {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    margin-bottom: 0.75rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
.sidebar-info-card p {
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.4c0 !important;
}
.sidebar-info-card .formula-box {
    background: rgba(8, 12, 33, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    font-family: monospace !important;
    font-size: 0.75rem !important;
    color: #38bdf8 !important;
    text-align: center !important;
    margin: 0.5rem 0 !important;
    line-height: 1.3 !important;
}
/* Buttons custom overrides */
div[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%) !important;
    color: white !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}
div[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(13, 18, 43, 0.6) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
div[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
}
.sidebar-footer {
    font-size: 0.75rem !important;
    color: #64748b !important;
    text-align: center !important;
    margin-top: 1.5rem !important;
    line-height: 1.6 !important;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}

/* ===== CUSTOM SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #060814; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #1e293b 0%, #313b7c 100%);
    border-radius: 4px;
}
</style>""", unsafe_allow_html=True)

# ============================================================
# ADAPTIVE THRESHOLD LOGIC
# ============================================================
def calculate_adaptive_threshold(re_errors, window_size=50, k=3.0):
    """
    Calculates a dynamic threshold based on a rolling window.
    Threshold = Rolling Mean + (k * Rolling Std Dev)
    This adapts to gradual changes in normal network behavior (concept drift).
    """
    df = pd.DataFrame({'re': re_errors})
    df['rolling_mean'] = df['re'].rolling(window=window_size, min_periods=1).mean()
    df['rolling_std'] = df['re'].rolling(window=window_size, min_periods=1).std().fillna(0)
    df['threshold'] = df['rolling_mean'] + (k * df['rolling_std'])
    
    # Anomaly is flagged if current RE > threshold
    df['is_anomaly'] = df['re'] > df['threshold']
    return df

# ============================================================
# LOAD SYNTHETIC N-BaIoT DATA
# ============================================================
import os

SYNTHETIC_FILE = "synthetic_nbaiot_features.csv"    

# Load the synthetic dataset once at startup
if os.path.exists(SYNTHETIC_FILE):
    print(f"[*] Loading synthetic data from {SYNTHETIC_FILE}...")
    SYNTHETIC_DATA = pd.read_csv(SYNTHETIC_FILE)
    BENIGN_SAMPLES = SYNTHETIC_DATA[SYNTHETIC_DATA['label'] == 'benign']
    ATTACK_SAMPLES = SYNTHETIC_DATA[SYNTHETIC_DATA['label'] == 'attack']
    DATA_LOADED = True
else:
    DATA_LOADED = False

def load_device_simulation_data(device_folder):
    if not os.path.exists(device_folder):
        return pd.DataFrame(), pd.DataFrame()
    benign_path = os.path.join(device_folder, "benign_traffic.csv")
    if not os.path.exists(benign_path):
        return pd.DataFrame(), pd.DataFrame()
    benign_df = pd.read_csv(benign_path, nrows=5000)
    attack_pieces = []
    for sub in ["gafgyt_attacks", "mirai_attacks"]:
        sub_dir = os.path.join(device_folder, sub)
        if os.path.exists(sub_dir):
            for file in os.listdir(sub_dir):
                if file.endswith(".csv"):
                    try:
                        adf = pd.read_csv(os.path.join(sub_dir, file), nrows=1000)
                        attack_pieces.append(adf)
                    except Exception:
                        pass
    attack_df = pd.concat(attack_pieces, ignore_index=True) if attack_pieces else pd.DataFrame()
    return benign_df, attack_df

def generate_live_batch(batch_size=20):
    benign_samples = st.session_state.get('benign_samples', pd.DataFrame())
    attack_samples = st.session_state.get('attack_samples', pd.DataFrame())
    if benign_samples.empty or attack_samples.empty:
        if DATA_LOADED:
            features_df_b = BENIGN_SAMPLES.drop(['label', 'Window_Size'], axis=1, errors='ignore')
            features_df_a = ATTACK_SAMPLES.drop(['label', 'Window_Size'], axis=1, errors='ignore')
        else:
            cols = [f"feat_{i}" for i in range(115)]
            features_df_b = pd.DataFrame(np.random.normal(0.1, 0.05, (100, 115)), columns=cols)
            features_df_a = pd.DataFrame(np.random.normal(0.8, 0.2, (100, 115)), columns=cols)
    else:
        features_df_b = benign_samples
        features_df_a = attack_samples
    n_benign = int(batch_size * 0.85)
    n_attack = batch_size - n_benign
    benign_batch = features_df_b.sample(n=n_benign, replace=True) if len(features_df_b) > 0 else pd.DataFrame()
    attack_batch = features_df_a.sample(n=n_attack, replace=True) if len(features_df_a) > 0 else pd.DataFrame()
    batch_df = pd.concat([benign_batch, attack_batch]).sample(frac=1)
    features = batch_df.values
    if st.session_state.scaler is not None:
        try:
            if features.shape[1] == st.session_state.scaler.n_features_in_:
                scaled_features = st.session_state.scaler.transform(features)
            else:
                scaled_features = features
        except Exception:
            scaled_features = features
    else:
        scaled_features = features
    if st.session_state.model is not None:
        try:
            reconstructed = st.session_state.model.predict(scaled_features, verbose=0)
            re_errors = np.mean(np.square(scaled_features - reconstructed), axis=1)
        except Exception:
            re_errors = np.mean(np.square(scaled_features), axis=1) * 0.1
    else:
        re_errors = [np.random.normal(0.001, 0.0002) if np.random.rand() > 0.15 else np.random.normal(0.35, 0.1) for _ in range(batch_size)]
    timestamps = [datetime.now().strftime("%H:%M:%S") for _ in range(len(re_errors))]
    return timestamps, re_errors


# ============================================================
# HELPER UI FUNCTIONS FOR REDESIGNED COMPONENTS
# ============================================================
def generate_sparkline_svg(values, color):
    if len(values) < 2:
        return f"""
        <svg width="100%" height="30" viewBox="0 0 100 30" preserveAspectRatio="none">
            <line x1="0" y1="15" x2="100" y2="15" stroke="{color}" stroke-dasharray="2,2" stroke-width="1"/>
        </svg>
        """
    import numpy as np
    vals = np.array(values)
    min_v = float(vals.min())
    max_v = float(vals.max())
    r = max_v - min_v
    if r == 0:
        r = 1.0
    
    width = 180
    height = 30
    points = []
    for i, val in enumerate(vals):
        x = (i / (len(vals) - 1)) * width
        y = height - 2 - ((val - min_v) / r) * (height - 4)
        points.append(f"{x:.1f},{y:.1f}")
        
    path_d = "M " + " L ".join(points)
    gradient_id = f"sparkline-grad-{hash(color) % 100000}"
    
    return f"""
    <svg width="100%" height="30" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="display: block;">
        <defs>
            <linearGradient id="{gradient_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="{color}" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="{path_d} L {width},{height} L 0,{height} Z" fill="url(#{gradient_id})" />
        <path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """

def draw_header_ui(placeholder):
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%B %d, %Y") if str(now.year) == "2025" else "May 6, 2025" # Keep date in screenshot format or close
    if now.year > 2025: # Dynamic date display in standard layout
         date_str = now.strftime("%b %d, %Y")
         
    satellite_svg = """<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M17 9A6 6 0 0 0 7 19" />
        <path d="M7 9a6 6 0 0 1 10 10" />
        <path d="M12 19v3M9 22h6" />
        <path d="M19 5l-5 5" />
        <circle cx="20" cy="4" r="1.5" fill="currentColor" />
        <path d="M14 3a9 9 0 0 1 7 7" />
        <path d="M16 6a5 5 0 0 1 2.5 2.5" />
    </svg>"""
    
    header_html = f"""
    <div class="main-header">
        <div class="header-left">
            <div class="header-icon-container">{satellite_svg}</div>
            <div class="header-titles">
                <h2><span style="color: #38bdf8;">N-BaloT Autoencoder:</span> Adaptive Anomaly Detection</h2>
                <p>Real-time Reconstruction Error Monitoring with Dynamic Thresholding</p>
            </div>
        </div>
        <div class="header-right">
            <div class="live-clock-badge">
                <div class="live-pill">
                    <span class="live-dot"></span> LIVE
                </div>
                <div class="time-display">{time_str}</div>
            </div>
            <div class="date-display">{date_str}</div>
        </div>
    </div>
    """
    placeholder.markdown(header_html, unsafe_allow_html=True)

def draw_metrics_ui(df, placeholders):
    if df.empty:
        total_flows = 0
        total_anomalies = 0
        anomaly_rate = 0.0
        current_threshold = 0.0
        spark_flows = []
        spark_anom = []
        spark_rate = []
        spark_thresh = []
    else:
        total_flows = len(df)
        total_anomalies = int(df['is_anomaly'].sum())
        anomaly_rate = (total_anomalies / total_flows) * 100
        current_threshold = float(df.iloc[-1]['threshold'])
        
        # Trends data
        spark_flows = df['re'].tail(45).values
        spark_anom = [float(val) if is_anom else float(df['re'].mean()) * 0.1 for val, is_anom in zip(df['re'].tail(45), df['is_anomaly'].tail(45))]
        spark_rate = df['is_anomaly'].rolling(20, min_periods=1).mean().tail(45).values * 100
        spark_thresh = df['threshold'].tail(45).values

    # Card 1: Flows Analyzed
    flows_icon = """<svg stroke="currentColor" fill="none" stroke-width="2.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg">
        <line x1="4" y1="21" x2="4" y2="14"></line>
        <line x1="4" y1="10" x2="4" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12" y2="3"></line>
        <line x1="20" y1="21" x2="20" y2="16"></line>
        <line x1="20" y1="12" x2="20" y2="3"></line>
        <line x1="1" y1="14" x2="7" y2="14"></line>
        <line x1="9" y1="8" x2="15" y2="8"></line>
        <line x1="17" y1="16" x2="23" y2="16"></line>
    </svg>"""
    spark_flows_svg = generate_sparkline_svg(spark_flows, "#38bdf8")
    placeholders[0].markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-top">
            <div class="metric-icon-container blue">{flows_icon}</div>
            <div class="metric-text-container">
                <div class="metric-value">{total_flows:,}</div>
                <div class="metric-label">Flows Analyzed</div>
            </div>
        </div>
        <div class="metric-sparkline">{spark_flows_svg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Anomalies Detected
    anom_icon = """<svg stroke="currentColor" fill="none" stroke-width="2.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>"""
    spark_anom_svg = generate_sparkline_svg(spark_anom, "#ef4444")
    placeholders[1].markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-top">
            <div class="metric-icon-container red">{anom_icon}</div>
            <div class="metric-text-container">
                <div class="metric-value red-text">{total_anomalies:,}</div>
                <div class="metric-label">Anomalies Detected</div>
            </div>
        </div>
        <div class="metric-sparkline">{spark_anom_svg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Anomaly Rate
    rate_icon = """<svg stroke="currentColor" fill="none" stroke-width="2.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg">
        <path d="M2 17a10 10 0 1 1 20 0"></path>
        <path d="m12 14 5-5"></path>
        <circle cx="12" cy="14" r="2"></circle>
    </svg>"""
    spark_rate_svg = generate_sparkline_svg(spark_rate, "#c084fc")
    placeholders[2].markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-top">
            <div class="metric-icon-container purple">{rate_icon}</div>
            <div class="metric-text-container">
                <div class="metric-value">{anomaly_rate:.2f}%</div>
                <div class="metric-label">Anomaly Rate</div>
            </div>
        </div>
        <div class="metric-sparkline">{spark_rate_svg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Current Threshold
    thresh_icon = """<svg stroke="currentColor" fill="none" stroke-width="2.5" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
    </svg>"""
    spark_thresh_svg = generate_sparkline_svg(spark_thresh, "#fb923c")
    placeholders[3].markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-top">
            <div class="metric-icon-container orange">{thresh_icon}</div>
            <div class="metric-text-container">
                <div class="metric-value orange-text">{current_threshold:.3f}</div>
                <div class="metric-label">Current Threshold</div>
            </div>
        </div>
        <div class="metric-sparkline">{spark_thresh_svg}</div>
    </div>
    """, unsafe_allow_html=True)

def draw_middle_chart_ui(df_display, placeholder):
    if df_display.empty:
        return
        
    df_normal = df_display[df_display['is_anomaly'] == False]
    df_anomalies = df_display[df_display['is_anomaly'] == True]
    
    fig = go.Figure()
    
    # 1. Normal traffic (Cyan/Blue)
    fig.add_trace(go.Scatter(
        x=df_normal.index, 
        y=df_normal['re'],
        mode='lines+markers',
        name='Normal',
        line=dict(color='#00d4ff', width=1.5),
        marker=dict(size=4, color='#00d4ff'),
        customdata=df_normal['timestamp'],
        hovertemplate="Flow Index: %{x}<br>Time: %{customdata}<br>Error: %{y:.4f}<extra></extra>"
    ))
    
    # 2. Anomaly markers (Red dots)
    fig.add_trace(go.Scatter(
        x=df_anomalies.index, 
        y=df_anomalies['re'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='#ef4444', size=8, symbol='circle'),
        customdata=df_anomalies['timestamp'],
        hovertemplate="Flow Index: %{x}<br>Time: %{customdata}<br>Error: %{y:.4f}<extra></extra>"
    ))
    
    # 3. Adaptive Threshold Line (Orange dashed)
    fig.add_trace(go.Scatter(
        x=df_display.index, 
        y=df_display['threshold'],
        mode='lines',
        name='Adaptive Threshold',
        line=dict(color='#f59e0b', width=1.5, dash='dash'),
        hovertemplate="Flow Index: %{x}<br>Threshold: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        height=380,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=40, l=40, r=10),
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(color='#94a3b8', size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            title=dict(text="Time (Flow Index)", font=dict(color='#94a3b8')),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#64748b'),
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Reconstruction Error (MSE)", font=dict(color='#94a3b8')),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#64748b'),
            zeroline=False
        )
    )
    
    with placeholder:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"main_chart_{st.session_state.loop_counter}")

def draw_bottom_row_ui(df, donut_placeholder, hist_placeholder, gauge_placeholder):
    # Calculations
    total_flows = len(df)
    total_anomalies = int(df['is_anomaly'].sum()) if not df.empty else 0
    anomaly_rate = (total_anomalies / total_flows * 100) if total_flows > 0 else 0.0
    current_threshold = float(df.iloc[-1]['threshold']) if not df.empty else 0.0
    
    # Common card style for Plotly charts
    card_shape = [dict(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(49, 59, 124, 0.4)", width=2)
    )]
    card_bg = '#0d122b'
    card_title_font = dict(color='#ffffff', size=16, family='Inter')
    
    # ==================== 1. DONUT CARD (Anomaly Breakdown) ====================
    if total_anomalies > 0 and not df.empty:
        anoms_df = df[df['is_anomaly'] == True]
        if not anoms_df.empty:
            mean_re = anoms_df['re'].mean()
            high_sev = int((anoms_df['re'] > mean_re * 1.5).sum())
            med_sev = int(((anoms_df['re'] > mean_re) & (anoms_df['re'] <= mean_re * 1.5)).sum())
            low_sev = max(0, total_anomalies - high_sev - med_sev)
            
            donut_labels = ['High Severity', 'Medium Severity', 'Low Severity']
            donut_values = [high_sev, med_sev, low_sev]
            donut_colors = ['#e11d48', '#f97316', '#fbbf24']
        else:
            high_sev = med_sev = low_sev = 0
            donut_labels = ['No Data']
            donut_values = [1]
            donut_colors = ['#1e293b']
    else:
        high_sev = med_sev = low_sev = 0
        donut_labels = ['No Anomalies']
        donut_values = [1]
        donut_colors = ['#1e293b']
    
    high_pct = (high_sev / total_anomalies * 100) if total_anomalies > 0 else 0.0
    med_pct = (med_sev / total_anomalies * 100) if total_anomalies > 0 else 0.0
    low_pct = (low_sev / total_anomalies * 100) if total_anomalies > 0 else 0.0
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=[f"High Severity ({high_pct:.1f}%)", f"Medium Severity ({med_pct:.1f}%)", f"Low Severity ({low_pct:.1f}%)"],
        values=donut_values,
        hole=0.68,
        marker=dict(colors=donut_colors, line=dict(color='#0d122b', width=2)),
        textinfo='none',
        showlegend=True,
        hoverinfo='label+value'
    )])
    
    fig_donut.update_layout(
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        margin=dict(t=50, b=20, l=20, r=120),  # Extra right margin for legend
        height=280,
        shapes=card_shape,
        title=dict(text="Anomaly Breakdown", font=card_title_font, x=0.02, y=0.95, xanchor='left'),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(color='#94a3b8', size=11),
            bgcolor='rgba(0,0,0,0)'
        ),
        annotations=[dict(
            text=f"<span style='font-size: 24px; font-weight: 700; color: #ffffff;'>{total_anomalies}</span><br><span style='font-size: 12px; color: #64748b; font-weight: 500;'>Total</span>",
            x=0.5, y=0.5,
            showarrow=False,
            align='center'
        )] if total_anomalies > 0 else [dict(
            text=f"<span style='font-size: 16px; color: #64748b;'>No Data</span>",
            x=0.5, y=0.5,
            showarrow=False
        )]
    )
    
    with donut_placeholder:
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False}, key=f"donut_chart_{st.session_state.loop_counter}")
    
    # ==================== 2. HISTOGRAM CARD (Reconstruction Error Distribution) ====================
    fig_hist = go.Figure()
    
    if not df.empty and len(df) > 0:
        fig_hist.add_trace(go.Histogram(
            x=df['re'].values,
            nbinsx=40,
            marker=dict(
                color='rgba(168,85,247,0.45)',
                line=dict(
                    color='#a855f7',
                    width=1.5
                )
            ),
            hovertemplate="Error: %{x:.4f}<br>Count: %{y}<extra></extra>"
        ))
        
        fig_hist.add_vline(
            x=current_threshold, 
            line_dash="dash", 
            line_color="#fb923c", 
            line_width=2
        )
    else:
        fig_hist.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color='#64748b', size=14)
        )
    
    fig_hist.update_layout(
        height=280,
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        margin=dict(t=50, b=30, l=40, r=20),
        showlegend=False,
        shapes=card_shape,
        title=dict(text="Reconstruction Error Distribution", font=card_title_font, x=0.02, y=0.95, xanchor='left'),
        xaxis=dict(
            title=dict(
                text="Reconstruction Error",
                font=dict(color='#94a3b8', size=11)
            ),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            tickmode='array',
            tickvals=[0, 200, 400, 600],
            ticktext=['0', '200', '400', '600'],
            range=[0, 800],
            tickfont=dict(color='#64748b', size=10),
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Frequency", font=dict(color='#94a3b8', size=11)),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.03)',
            tickfont=dict(color='#64748b', size=10),
            zeroline=False
        )
    )
    
    with hist_placeholder:
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False}, key=f"hist_chart_{st.session_state.loop_counter}")
    
    # ==================== 3. GAUGE CARD (System Health) ====================
    if total_flows > 0:
        health_score = max(0.0, min(100.0, 100.0 - (anomaly_rate * 10.0)))
        if health_score >= 85:
            health_color = '#10b981'
            health_label = 'Excellent'
        elif health_score >= 60:
            health_color = '#f97316'
            health_label = 'Good'
        else:
            health_color = '#ef4444'
            health_label = 'Critical'
    else:
        health_score = 100.0
        health_color = '#10b981'
        health_label = 'Ready'
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=health_score,
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {
                'color': health_color,
                'thickness': 0.28
            },
            'bgcolor': 'rgba(255, 255, 255, 0.05)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [60, 85], 'color': 'rgba(249, 115, 22, 0.15)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
        }
    ))
    
    # Add text INSIDE the gauge arc using annotations
    fig_gauge.add_annotation(
        text=f"<span style='color:{health_color}; font-size:30px; font-weight:bold;'>{health_label}</span>",
        x=0.5,
        y=0.50,
        showarrow=False,
        xref="paper",
        yref="paper"
    )

    fig_gauge.add_annotation(
        text="<span style='color:#94a3b8; font-size:14px;'>System Status</span>",
        x=0.5,
        y=0.40,
        showarrow=False,
        xref="paper",
        yref="paper"
    )

    fig_gauge.add_annotation(
        text=f"<span style='color:{health_color}; font-size:22px;'>🛡️</span>",
        x=0.5,
        y=0.22,
        showarrow=False,
        xref="paper",
        yref="paper"
    )
    
    fig_gauge.update_layout(
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        margin=dict(t=50, b=20, l=20, r=20),
        height=280,
        shapes=card_shape,
        title=dict(text="System Health", font=card_title_font, x=0.02, y=0.95, xanchor='left')
    )
    
    with gauge_placeholder:
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False}, key=f"gauge_chart_{st.session_state.loop_counter}")

def draw_alert_log_ui(placeholder):
    """Renders the real-time alert log table."""
    # Use .get() to prevent AttributeError if alert_log isn't initialized yet
    if not st.session_state.get('alert_log', []):
        placeholder.markdown("""
        <div style="background: #0d122b; border: 1px solid rgba(49, 59, 124, 0.4); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #64748b; margin: 0;">🔒 No anomalies detected yet. System is monitoring...</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df_log = pd.DataFrame(st.session_state.alert_log)
    # Show newest alerts first
    df_log = df_log.iloc[::-1].reset_index(drop=True)
    
    # Custom styling for severity
    def color_severity(val):
        if val == 'High': return 'color: #ef4444; font-weight: bold;'
        if val == 'Medium': return 'color: #f97316; font-weight: bold;'
        return 'color: #fbbf24;'

    # FIX: Use 'map' for Pandas >= 2.1.0, fallback to 'applymap' for older versions
    if hasattr(df_log.style, 'map'):
        styled_df = df_log.style.map(color_severity, subset=['Severity'])
    else:
        styled_df = df_log.style.applymap(color_severity, subset=['Severity'])

    placeholder.markdown('<div class="chart-header-title" style="margin-bottom: 0.5rem;"> Live Alert Feed</div>', unsafe_allow_html=True)
    placeholder.dataframe(styled_df, use_container_width=True, height=180)

def draw_footer_ui(placeholder):
    brain_icon = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/>
        <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/>
    </svg>"""
    
    db_icon = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
        <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>
    </svg>"""
    
    signal_icon = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.59 16.11a6 6 0 0 1 6.82 0M12 20h.01"/>
    </svg>"""
    
    clock_icon = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <polyline points="12 6 12 12 16 14"/>
    </svg>"""
    
    # Dynamic values based on Model/Sniffer states
    is_model_active = "Active" if st.session_state.running else "Idle"
    model_status_class = "green" if st.session_state.running else "purple"
    is_model_loaded = "Loaded" if st.session_state.model is not None else "Not Found"
    model_load_class = "green" if st.session_state.model is not None else "red"
    data_source_mode = "Live Capture" if st.session_state.get('traffic_source', 'Simulated Traffic') == 'Live Packet Capture' else "Dataset Sim"
    data_source_class = "blue" if st.session_state.get('traffic_source', 'Simulated Traffic') == 'Live Packet Capture' else "purple"
    
    footer_html = f"""
    <div class="footer-bar">
        <div class="footer-item">
            <div class="footer-icon-div {model_status_class}">{brain_icon}</div>
            <div class="footer-text-group">
                <div class="footer-label">Model Status</div>
                <div class="footer-val {model_status_class}">{is_model_active}</div>
            </div>
        </div>
        <div class="footer-item">
            <div class="footer-icon-div {model_load_class}">{db_icon}</div>
            <div class="footer-text-group">
                <div class="footer-label">Autoencoder</div>
                <div class="footer-val {model_load_class}">{is_model_loaded}</div>
            </div>
        </div>
        <div class="footer-item">
            <div class="footer-icon-div {data_source_class}">{signal_icon}</div>
            <div class="footer-text-group">
                <div class="footer-label">Data Source</div>
                <div class="footer-val {data_source_class}">{data_source_mode}</div>
            </div>
        </div>
        <div class="footer-item">
            <div class="footer-icon-div purple">{clock_icon}</div>
            <div class="footer-text-group">
                <div class="footer-label">Last Update</div>
                <div class="footer-val purple">Just now</div>
            </div>
        </div>
    </div>
    """
    placeholder.markdown(footer_html, unsafe_allow_html=True)


# ============================================================
# SIDEBARA WIDGETS RENDERING
# ============================================================
with st.sidebar:
    st.markdown("### 📡 IoT Device Configuration")
    import os
    available_devices = [
        d for d in os.listdir(".") 
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "benign_traffic.csv"))
    ]
    if not available_devices:
        available_devices = ["Danmini_Doorbell"]
    selected_device = st.selectbox(
        "Select IoT Device Dataset",
        options=available_devices,
        index=0,
        help="Select which IoT device traffic directory to load and monitor."
    )
    if 'last_device' not in st.session_state or st.session_state.last_device != selected_device:
        st.session_state.last_device = selected_device
        b_df, a_df = load_device_simulation_data(selected_device)
        st.session_state.benign_samples = b_df
        st.session_state.attack_samples = a_df
        
    with st.expander("🛠️ Train Autoencoder on Device", expanded=False):
        st.markdown("<p style='font-size: 0.8rem; color: #94a3b8;'>Train a fresh autoencoder on the benign dataset (features: 115) from this device's folder.</p>", unsafe_allow_html=True)
        train_epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        train_samples = st.number_input("Max Training Samples", min_value=1000, max_value=100000, value=20000, step=5000)
        train_button = st.button("🚀 Start Training Mode", use_container_width=True)
        if train_button:
            progress_bar = st.empty()
            status_text = st.empty()
            def update_progress(epoch, total_epochs, loss, val_loss):
                progress_bar.progress(float(epoch) / float(total_epochs))
                status_text.markdown(f"**Epoch {epoch}/{total_epochs}**<br>Loss: `{loss:.5f}` | Val Loss: `{val_loss:.5f}`", unsafe_allow_html=True)
            try:
                from train_autoencoder import train_model as execute_train
                device_path = os.path.join(".", selected_device)
                execute_train(device_path, epochs=train_epochs, batch_size=256, max_samples=train_samples, progress_callback=update_progress)
                st.session_state.model = load_model('nbaiot_autoencoder.h5', compile=False)
                st.session_state.scaler = joblib.load('scaler.save')
                st.success("🎉 Model trained and loaded successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("### ⚙️ Detection Parameters")
    window_size = st.slider("Adaptive Window Size (N)", 10, 200, 50, 
                            help="Number of recent flows used to calculate baseline behavior")
    k_multiplier = st.slider("Sensitivity Multiplier (k)", 0.50, 10.00, 3.00, 0.25,
                             help="Threshold = Mean + (k * StdDev). Lower 'k' = more sensitive.")
    
    st.markdown("### 🎛️ Live sniffer / Simulation Control")
    traffic_source = st.selectbox(
        "Traffic Source Mode",
        options=["Simulated Traffic", "Live Packet Capture"],
        index=0,
        help="Simulate historical batch flows or tap into live local network packet cap."
    )
    st.session_state.traffic_source = traffic_source
    run_live = st.button("▶️ Start Live Stream", type="primary")
    stop_live = st.button("️⏹️ Stop")
    
    # >>> EXPORT BUTTON <<<
    st.markdown("---")
    if 'alert_log' in st.session_state and st.session_state.alert_log:
        df_export = pd.DataFrame(st.session_state.alert_log)
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Alert Log (CSV)",
            data=csv,
            file_name=f"anomaly_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.markdown("<p style='font-size: 0.8rem; color: #64748b; text-align: center;'>No alerts to export yet.</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info-card">
        <div class="info-title">ℹ️ How Adaptive Threshold Works</div>
        <p>Instead of a fixed line, the threshold dynamically adjusts to recent network behavior:</p>
        <div class="formula-box">Threshold = Rolling_Mean +<br>(k × Rolling_StdDev)</div>
        <p>This prevents false alarms during normal fluctuations and focuses on real anomalies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-footer">
        N-BaloT Autoencoder v2.0 <br>
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE & SYSTEM ACTION
# ============================================================
if 'model' not in st.session_state:
    st.session_state.model = load_trained_model()
if 'scaler' not in st.session_state:
    st.session_state.scaler = load_trained_scaler()
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=['timestamp', 're', 'threshold', 'is_anomaly'])
if 'running' not in st.session_state:
    st.session_state.running = False
if 'loop_counter' not in st.session_state:
    st.session_state.loop_counter = 0
if 'extractor' not in st.session_state:
    from kitsune_extractor import KitsuneExtractor
    st.session_state.extractor = KitsuneExtractor()
if 'sniffer' not in st.session_state:
    from packet_sniffer import BackgroundSniffer
    st.session_state.sniffer = BackgroundSniffer()
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []  

# TO CAPTURE THE START BUTTON CLICK
if run_live:
    st.session_state.running = True
    st.session_state.data_history = pd.DataFrame(columns=['timestamp', 're', 'threshold', 'is_anomaly'])
    if st.session_state.get('traffic_source', 'Simulated Traffic') == 'Live Packet Capture':
        st.session_state.sniffer.start()
    
if stop_live:
    st.session_state.running = False
    st.session_state.sniffer.stop()


# ============================================================
# PLACEHOLDERS DECLARATION FOR THE DASHBOARD GRID
# ============================================================
header_placeholder = st.empty()

# 4 Columns for Metrics
col1, col2, col3, col4 = st.columns(4)
metric_phs = [col1.empty(), col2.empty(), col3.empty(), col4.empty()]

# 1. Main chart card controls and container
st.markdown(' <div style="height: 10px;"></div>', unsafe_allow_html=True)
c_title, c_select = st.columns([3, 1])
with c_title:
    st.markdown('<div class="chart-header-title">Reconstruction Error vs. Adaptive Threshold</div>', unsafe_allow_html=True)
with c_select:
    points_to_show = st.selectbox("Points to Show", options=[100, 200, 500], index=2, label_visibility="collapsed")

# FIX: Wrap the chart placeholder in the static HTML card ONCE
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
chart_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# Bottom Row: 3 Analytics Cards - Clean layout without HTML wrappers
bottom_col1, bottom_col2, bottom_col3 = st.columns(3)

with bottom_col1:
    donut_ph = st.empty()

with bottom_col2:
    hist_ph = st.empty()

with bottom_col3:
    gauge_ph = st.empty()

alert_placeholder = st.empty()
footer_placeholder = st.empty()

# ============================================================
# EVENT LOOP / STREAM CONTROLLER
# ============================================================
# Define the fragment that runs every 0.5 seconds
@st.fragment(run_every=0.5)
def live_stream_fragment():
    if st.session_state.running:
        st.session_state.loop_counter += 1
        
        # 1. Sample features & run through Model
        if st.session_state.get('traffic_source', 'Simulated Traffic') == 'Live Packet Capture':
            import queue
            pkts = []
            for _ in range(15):
                try:
                    p = st.session_state.sniffer.packet_queue.get_nowait()
                    pkts.append(p)
                except queue.Empty:
                    break
            
            if not pkts:
                # Baseline error when network is idle
                timestamps = [datetime.now().strftime("%H:%M:%S")]
                re_errors = [float(np.random.normal(0.001, 0.0002))]
            else:
                timestamps = []
                features_list = []
                for p in pkts:
                    t_val, src_mac, src_ip, src_port, dst_ip, dst_port, size = p
                    feat = st.session_state.extractor.extract_features(t_val, src_mac, src_ip, src_port, dst_ip, dst_port, size)
                    features_list.append(feat)
                    timestamps.append(datetime.fromtimestamp(t_val).strftime("%H:%M:%S"))
                
                if st.session_state.model is not None and st.session_state.scaler is not None:
                    try:
                        scaled = st.session_state.scaler.transform(np.array(features_list))
                        reconstructed = st.session_state.model.predict(scaled, verbose=0)
                        re_errors = list(np.mean(np.square(scaled - reconstructed), axis=1))
                    except Exception:
                        re_errors = [0.001] * len(features_list)
                else:
                    re_errors = [0.001] * len(features_list)
        else:
            timestamps, re_errors = generate_live_batch(batch_size=15)
        
        # 2. Calculate adaptive threshold
        recent_re = list(st.session_state.data_history['re'].tail(window_size)) + list(re_errors)
        temp_df = calculate_adaptive_threshold(recent_re, window_size=window_size, k=k_multiplier)
        
        new_thresholds = temp_df['threshold'].tail(len(re_errors)).values
        new_anomalies = temp_df['is_anomaly'].tail(len(re_errors)).values
        
        new_data = pd.DataFrame({
            'timestamp': timestamps,
            're': re_errors,
            'threshold': new_thresholds,
            'is_anomaly': new_anomalies
        })
         
        # >>> TO LOG ANOMALIES <<<
        new_anomalies_df = new_data[new_data['is_anomaly'] == True]
        if not new_anomalies_df.empty:
            current_idx = len(st.session_state.data_history)
            hist_mean = st.session_state.data_history['re'].mean() if not st.session_state.data_history.empty else 0
            
            for i, (_, row) in enumerate(new_anomalies_df.iterrows()):
                severity = "Low"
                if row['re'] > hist_mean * 1.5:
                    severity = "High"
                elif row['re'] > hist_mean:
                    severity = "Medium"
                    
                st.session_state.alert_log.append({
                    'Time': row['timestamp'],
                    'Flow Index': current_idx + i + 1,
                    'Re Error': f"{row['re']:.4f}",
                    'Threshold': f"{row['threshold']:.4f}",
                    'Severity': severity
                })
                
            # Keep log size manageable (last 100 alerts)
            if len(st.session_state.alert_log) > 100:
                st.session_state.alert_log = st.session_state.alert_log[-100:]
        
        st.session_state.data_history = pd.concat([st.session_state.data_history, new_data], ignore_index=True)
        if len(st.session_state.data_history) > 500:
            st.session_state.data_history = st.session_state.data_history.tail(500).reset_index(drop=True)
            
        df = st.session_state.data_history
        df_display = df.tail(points_to_show)
        
        # 3. Render UI
        draw_header_ui(header_placeholder)
        draw_metrics_ui(df, metric_phs)
        draw_middle_chart_ui(df_display, chart_placeholder)
        draw_bottom_row_ui(df, donut_ph, hist_ph, gauge_ph)
        draw_alert_log_ui(alert_placeholder) 
        draw_footer_ui(footer_placeholder)

# Call the fragment
if st.session_state.running:
    live_stream_fragment()
else:
    # Stopped or initial state render
    draw_header_ui(header_placeholder)
    if st.session_state.data_history.empty:
        for i, ph in enumerate(metric_phs):
            lbls = ["Flows Analyzed", "Anomalies Detected", "Anomaly Rate", "Current Threshold"]
            ph.markdown(f"""
            <div class="metric-card-custom">
                <div class="metric-top">
                    <div style="font-size: 1.2rem; color: #64748b;">-</div>
                    <div class="metric-text-container">
                        <div class="metric-value">-</div>
                        <div class="metric-label">{lbls[i]}</div>
                    </div>
                </div>
                <div class="metric-sparkline"></div>
            </div>
            """, unsafe_allow_html=True)
        chart_placeholder.info("Click '▶️ Start Live Stream' in the sidebar to begin simulation.")
        draw_bottom_row_ui(pd.DataFrame(), donut_ph, hist_ph, gauge_ph)
    else:
        df = st.session_state.data_history
        df_display = df.tail(points_to_show)
        draw_metrics_ui(df, metric_phs)
        draw_middle_chart_ui(df_display, chart_placeholder)
        draw_bottom_row_ui(df, donut_ph, hist_ph, gauge_ph)
        draw_alert_log_ui(alert_placeholder)
        draw_footer_ui(footer_placeholder)
