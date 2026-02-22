"""
MoodSense: Music Emotion Classification Platform
─────────────────────────────────────────────────
MS DSP 422 – Practical Machine Learning | Group 3

Repo layout:
    moodsense-app/
    ├── moodsense_app.py   ← this file (entry point)
    ├── requirements.txt
    └── model/
        ├── audio_scaler.pkl
        ├── vader_scaler.pkl   (optional)
        ├── tfidf.pkl
        ├── label_encoder.pkl
        ├── lightgbm_model.pkl
        ├── xgboost_model.pkl
        ├── ensemble_model.pkl
        ├── linearsvc_model.pkl
        ├── logistic_model.pkl
        └── model_metadata.json

NO dataset file is required or shipped.
All features run purely from trained model artifacts + user input.

Run locally:
    pip install -r requirements.txt
    streamlit run moodsense_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import re
import json
import warnings
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# WeightedEnsemble  — MUST live before any pickle.load()
# ══════════════════════════════════════════════════════════════════════════════
class WeightedEnsemble:
    """Weighted soft-voting ensemble (LightGBM + XGBoost + LogReg)."""

    def __init__(self, models: dict, weights: dict, label_encoder):
        self.models        = models
        self.weights       = weights
        self.label_encoder = label_encoder

    def predict_proba(self, X):
        total  = sum(self.weights.values())
        probas = [m.predict_proba(X) * (self.weights[n] / total)
                  for n, m in self.models.items()]
        return np.sum(probas, axis=0)

    def predict(self, X):
        return self.label_encoder.inverse_transform(
            np.argmax(self.predict_proba(X), axis=1)
        )


# ══════════════════════════════════════════════════════════════════════════════
# PATHS — all relative to this file (repo root)
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = _ROOT / "model"

AUDIO_SCALER_PATH  = MODEL_DIR / "audio_scaler.pkl"
VADER_SCALER_PATH  = MODEL_DIR / "vader_scaler.pkl"   # optional
TFIDF_PATH         = MODEL_DIR / "tfidf.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH      = MODEL_DIR / "model_metadata.json"

MODEL_FILES = {
    "LightGBM":  MODEL_DIR / "lightgbm_model.pkl",
    "XGBoost":   MODEL_DIR / "xgboost_model.pkl",
    "Ensemble":  MODEL_DIR / "ensemble_model.pkl",
    "LinearSVC": MODEL_DIR / "linearsvc_model.pkl",
    "LogReg":    MODEL_DIR / "logistic_model.pkl",
}


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_STYLE = {
    "happy": {"color": "#10b981", "emoji": "😊", "desc": "Joyful and uplifting"},
    "sad":   {"color": "#3b82f6", "emoji": "😢", "desc": "Melancholic and reflective"},
    "anger": {"color": "#ef4444", "emoji": "🔥", "desc": "Intense and powerful"},
    "love":  {"color": "#ec4899", "emoji": "💕", "desc": "Romantic and tender"},
}

# Feature column order MUST match training exactly
PRIMARY_AUDIO_COLS = [
    "Energy", "Danceability", "Positiveness", "Speechiness",
    "Liveness", "Acousticness", "Instrumentalness", "Tempo",
    "Loudness", "Popularity",
]
API_AUDIO_COLS = [
    "api_valence", "api_danceability", "api_energy", "api_acousticness",
    "api_instrumentalness", "api_speechiness", "api_liveness",
    "api_tempo", "api_loudness", "api_duration_ms", "api_mode", "api_key",
]

# Valid ranges for each audio feature (lo, hi, default)
AUDIO_RANGES = {
    "Energy":           (0.0,  1.0,  0.65),
    "Danceability":     (0.0,  1.0,  0.60),
    "Positiveness":     (0.0,  1.0,  0.50),
    "Speechiness":      (0.0,  1.0,  0.05),
    "Liveness":         (0.0,  1.0,  0.12),
    "Acousticness":     (0.0,  1.0,  0.30),
    "Instrumentalness": (0.0,  1.0,  0.10),
    "Tempo":            (60,   200,  120),
    "Loudness":         (-60,  0,    -7),
    "Popularity":       (0,    100,  50),
}

# Per-emotion audio centroids (overridden by model_metadata.json if present)
EMOTION_CENTROIDS_FALLBACK = {
    "happy": {"Energy":0.75,"Danceability":0.72,"Positiveness":0.78,"Speechiness":0.07,
              "Liveness":0.17,"Acousticness":0.22,"Instrumentalness":0.04,
              "Tempo":128,"Loudness":-5,"Popularity":65},
    "sad":   {"Energy":0.38,"Danceability":0.42,"Positiveness":0.25,"Speechiness":0.04,
              "Liveness":0.12,"Acousticness":0.60,"Instrumentalness":0.08,
              "Tempo":98,"Loudness":-10,"Popularity":48},
    "anger": {"Energy":0.82,"Danceability":0.55,"Positiveness":0.35,"Speechiness":0.12,
              "Liveness":0.18,"Acousticness":0.15,"Instrumentalness":0.05,
              "Tempo":145,"Loudness":-4,"Popularity":55},
    "love":  {"Energy":0.52,"Danceability":0.60,"Positiveness":0.72,"Speechiness":0.05,
              "Liveness":0.10,"Acousticness":0.40,"Instrumentalness":0.03,
              "Tempo":108,"Loudness":-7,"Popularity":60},
}

# Pre-designed playlist presets — each is just a prompt fed to the ML pipeline
PRESET_PLAYLISTS = [
    {"id":"work_focus",      "icon":"💼","title":"Deep Work Focus",
     "subtitle":"Instrumental · low distraction · steady tempo",
     "prompt":"instrumental focus concentration work study calm steady beats no lyrics","n":20},
    {"id":"rainy_afternoon", "icon":"🌧️","title":"Rainy Afternoon",
     "subtitle":"Melancholic · acoustic · introspective",
     "prompt":"rain melancholic quiet acoustic reflective slow afternoon sad gentle","n":18},
    {"id":"heartbreak",      "icon":"💔","title":"Heartbreak Hotel",
     "subtitle":"Emotional · raw · post-breakup vibes",
     "prompt":"heartbreak crying pain love lost breakup emotional tears missing someone","n":18},
    {"id":"late_night_drive","icon":"🚗","title":"Late Night Drive",
     "subtitle":"Dark · atmospheric · driving energy",
     "prompt":"night dark driving highway electric atmospheric moody intensity beat","n":20},
    {"id":"summer_party",    "icon":"🌞","title":"Summer Party",
     "subtitle":"High energy · danceable · joyful",
     "prompt":"party dance happy upbeat energy summer fun celebration joyful loud","n":20},
    {"id":"romance",         "icon":"🌹","title":"Date Night",
     "subtitle":"Romantic · tender · warm",
     "prompt":"love romance tender sweet gentle warmth affection intimate caring","n":18},
    {"id":"anger_release",   "icon":"⚡","title":"Rage Release",
     "subtitle":"Intense · powerful · cathartic",
     "prompt":"anger intense powerful aggressive rage frustration electric loud cathartic","n":18},
    {"id":"morning_run",     "icon":"🏃","title":"Morning Run",
     "subtitle":"Energetic · motivating · fast tempo",
     "prompt":"run exercise morning energetic motivating fast beats workout pump adrenaline","n":20},
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoodSense | Music Emotion Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, body { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

:root {
    --ink:#0f172a; --muted:#64748b; --border:#e2e8f0;
    --bg:#f8fafc;  --card:#ffffff;
    --blue:#1d4ed8; --blue-lt:#3b82f6;
}
.main { background:var(--bg); }

.hero {
    background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 60%,#3b82f6 100%);
    padding:3.5rem 2.5rem; border-radius:20px; color:white;
    text-align:center; margin-bottom:2rem;
    box-shadow:0 24px 64px rgba(29,78,216,.25);
    position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-title { font-size:3.5rem; font-weight:800; margin:0 0 1rem; letter-spacing:-.03em; }
.hero-sub   { font-size:1.2rem; opacity:.85; margin:0; }

.kpi {
    background:var(--card); padding:1.5rem; border-radius:14px;
    border:1px solid var(--border); box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.kpi-val { font-size:2.2rem; font-weight:800; color:var(--blue); margin:0; font-family:'Syne',sans-serif; }
.kpi-lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase;
           letter-spacing:.08em; font-weight:600; margin-top:.4rem; }

.emo-card  { padding:2.5rem; border-radius:18px; text-align:center;
             box-shadow:0 12px 40px rgba(0,0,0,.18); margin:1.5rem 0; }
.emo-emoji { font-size:5rem; }
.emo-name  { font-size:2.2rem; font-weight:800; margin:.8rem 0 .4rem; font-family:'Syne',sans-serif; }
.emo-desc  { font-size:1rem; opacity:.85; }

.atlas-card {
    background:var(--card); border-radius:16px; padding:1.5rem;
    border:1px solid var(--border); box-shadow:0 4px 16px rgba(0,0,0,.06);
    margin-bottom:1rem;
}
.atlas-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; margin:0 0 .3rem; }
.atlas-sub   { font-size:.82rem; color:var(--muted); margin-bottom:.8rem; }

.batch-row {
    display:flex; align-items:center; gap:1rem; padding:.65rem .5rem;
    border-radius:8px; border-bottom:1px solid var(--border); transition:background .15s;
}
.batch-row:hover { background:var(--bg); }
.batch-num  { width:1.8rem; text-align:right; color:var(--muted); font-size:.85rem; font-weight:600; flex-shrink:0; }
.batch-info { flex:1; min-width:0; }
.batch-name { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.batch-artist { font-size:.8rem; color:var(--muted); }
.badge { font-size:.72rem; font-weight:700; padding:.2rem .65rem; border-radius:999px; flex-shrink:0; }
.conf-wrap { width:80px; flex-shrink:0; }
.conf-bg   { background:#e2e8f0; border-radius:4px; height:6px; }
.conf-fg   { border-radius:4px; height:6px; }

.playlist-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:16px; padding:1.75rem;
    box-shadow:0 4px 16px rgba(0,0,0,.06); margin-bottom:1rem;
}
.playlist-header { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; margin:0 0 .3rem; }
.playlist-meta   { color:var(--muted); font-size:.85rem; margin-bottom:1rem; }
.track-row {
    display:flex; align-items:center; gap:1rem; padding:.65rem .5rem;
    border-radius:8px; border-bottom:1px solid var(--border); transition:background .15s;
}
.track-row:hover  { background:var(--bg); }
.track-num   { width:1.8rem; text-align:right; color:var(--muted); font-size:.85rem; font-weight:600; flex-shrink:0; }
.track-info  { flex:1; min-width:0; }
.track-name  { font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.track-artist{ font-size:.8rem; color:var(--muted); }

[data-testid="stSidebar"] { background:linear-gradient(180deg,#0f172a,#1e293b); }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }

.stButton button {
    background:linear-gradient(135deg,var(--blue),var(--blue-lt));
    color:white; border:none; border-radius:10px;
    padding:.75rem 2rem; font-weight:600; letter-spacing:.02em; transition:all .25s;
}
.stButton button:hover { box-shadow:0 8px 24px rgba(29,78,216,.35); transform:translateY(-2px); }

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML models…")
def load_models() -> dict | None:
    """Load all pkl artifacts from model/. Returns bundle dict or None."""
    bundle: dict = {}

    def _load(p: Path):
        with open(p, "rb") as f:
            return pickle.load(f)

    missing = [str(p) for p in [AUDIO_SCALER_PATH, TFIDF_PATH, LABEL_ENCODER_PATH]
               if not p.exists()]
    if missing:
        st.error("❌ Required model files missing:\n" +
                 "\n".join(f"  • {m}" for m in missing))
        return None

    try:
        bundle["audio_scaler"]  = _load(AUDIO_SCALER_PATH)
        bundle["tfidf"]         = _load(TFIDF_PATH)
        bundle["label_encoder"] = _load(LABEL_ENCODER_PATH)
        if VADER_SCALER_PATH.exists():
            bundle["vader_scaler"] = _load(VADER_SCALER_PATH)
        for name, path in MODEL_FILES.items():
            if path.exists():
                bundle[name] = _load(path)
        return bundle
    except Exception as exc:
        st.error(f"❌ Model loading failed: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE — mirrors training exactly, no dataset needed
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\b(la|na|oh|ah|uh|yeah|ooh|mm|hey)\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def audio_vals_to_vector(vals: dict) -> list:
    """
    Convert audio feature dict → 29-d vector matching training column order:
        10 primary + 12 API (zeros at inference) + 7 engineered
    """
    def sv(k, d=0.0): return float(vals.get(k, d))

    energy   = sv("Energy",           0.5)
    dance    = sv("Danceability",      0.5)
    pos      = sv("Positiveness",      0.5)
    speech   = sv("Speechiness",       0.05)
    liveness = sv("Liveness",          0.1)
    acoustic = sv("Acousticness",      0.5)
    instr    = sv("Instrumentalness",  0.1)
    tempo    = sv("Tempo",           120.0)
    loudness = sv("Loudness",         -7.0)
    pop      = sv("Popularity",       50.0)

    primary    = [energy, dance, pos, speech, liveness, acoustic, instr, tempo, loudness, pop]
    api_zeros  = [0.0] * len(API_AUDIO_COLS)
    engineered = [
        dance * pos / 100,       # feel_good
        energy - acoustic,       # electronic
        tempo * energy / 100,    # intensity
        speech / (instr + 1),    # vocal_dom
        pos / (energy + 1),      # mood_energy
        0.0,                     # valence_dance (API unavailable)
        0.0,                     # arousal       (API unavailable)
    ]
    return primary + api_zeros + engineered


def build_feature_matrix(audio_rows: list, lyrics_list: list, bundle: dict):
    """
    Build the sparse feature matrix expected by trained models.
    Shape: (n, 25033)  — TF-IDF(25000) + VADER(4) + Audio(29)
    No dataset access — only user-supplied values.
    """
    n       = len(audio_rows)
    tfidf   = bundle.get("tfidf")
    n_tfidf = tfidf.get_feature_names_out().shape[0] if tfidf else 25_000

    # TF-IDF
    if tfidf:
        cleaned = [clean_text(lyr) if str(lyr).strip() else "" for lyr in lyrics_list]
        X_tfidf = tfidf.transform(cleaned)
    else:
        X_tfidf = csr_matrix((n, n_tfidf))

    # VADER
    vader_obj = SentimentIntensityAnalyzer()
    vader_mat = np.zeros((n, 4), dtype="float32")
    for i, lyr in enumerate(lyrics_list):
        if str(lyr).strip():
            vs = vader_obj.polarity_scores(clean_text(lyr))
            vader_mat[i] = [vs["neg"], vs["neu"], vs["pos"], vs["compound"]]
    if "vader_scaler" in bundle:
        try:
            vader_mat = bundle["vader_scaler"].transform(vader_mat).astype("float32")
        except Exception:
            pass

    # Audio
    audio_mat = np.array([audio_vals_to_vector(r) for r in audio_rows], dtype="float32")
    scaler    = bundle.get("audio_scaler")
    if scaler:
        try:
            audio_mat = scaler.transform(audio_mat)
        except Exception:
            pass

    return hstack([X_tfidf, csr_matrix(vader_mat), csr_matrix(audio_mat)])


def run_inference(X, bundle: dict, model_name: str):
    """
    Run model inference.
    Returns (predicted_string_labels: np.ndarray, proba_matrix: np.ndarray)
    """
    model = bundle.get(model_name)
    le    = bundle.get("label_encoder")
    if model is None:
        raise ValueError(f"Model '{model_name}' not in bundle.")

    if hasattr(model, "predict_proba"):
        proba    = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        classes  = le.classes_ if le is not None else list(EMOTION_STYLE.keys())
        # Decode: le.classes_ may be strings or ints
        if le is not None:
            labels = np.array([str(le.inverse_transform([i])[0]) for i in pred_idx])
        else:
            labels = np.array([str(classes[i]) for i in pred_idx])
    else:
        # LinearSVC — no predict_proba
        raw    = model.predict(X)
        labels = np.array([str(r) for r in raw])
        cls    = list(EMOTION_STYLE.keys())
        proba  = np.zeros((len(labels), len(cls)), dtype="float32")
        for i, lbl in enumerate(labels):
            if lbl in cls:
                proba[i, cls.index(lbl)] = 1.0

    return labels, proba


# ══════════════════════════════════════════════════════════════════════════════
# PLAYLIST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def parse_prompt_to_weights(prompt: str, bundle: dict) -> dict:
    """
    NLP: free-text → soft emotion weight dict via
         VADER sentiment + TF-IDF cosine similarity.
    Purely continuous — no threshold rules.
    """
    vader_obj = SentimentIntensityAnalyzer()
    vs        = vader_obj.polarity_scores(prompt)
    compound, neg, pos = vs["compound"], vs["neg"], vs["pos"]

    vader_sig = {
        "happy": pos * (1 + compound) / 2,
        "love":  pos * max(0.0, 1 - neg),
        "sad":   neg * (2 - compound) / 2,
        "anger": neg * (1 + abs(compound)),
    }

    EMOTION_DOCS = {
        "happy": "joy dance smile laugh celebrate fun sunshine bright energy cheerful upbeat",
        "sad":   "cry tears sorrow grief lonely pain heartbreak quiet dark rain melancholy",
        "anger": "rage fury fight anger aggressive loud power electric intense frustration scream",
        "love":  "love tender romance sweet kiss warmth affection gentle care heart beloved",
    }
    tfidf_sig = {e: 0.0 for e in EMOTION_STYLE}
    tfidf     = bundle.get("tfidf")
    if tfidf:
        try:
            p_vec = tfidf.transform([clean_text(prompt)])
            for em, doc in EMOTION_DOCS.items():
                d_vec = tfidf.transform([clean_text(doc)])
                tfidf_sig[em] = float(cosine_similarity(p_vec, d_vec)[0][0])
        except Exception:
            pass

    raw   = {em: 0.5 * vader_sig.get(em, 0.0) + 0.5 * tfidf_sig[em] for em in EMOTION_STYLE}
    total = sum(raw.values()) or 1.0
    return {em: v / total for em, v in raw.items()}


def get_centroids(metadata: dict) -> dict:
    return metadata.get("emotion_audio_centroids") or EMOTION_CENTROIDS_FALLBACK


def generate_playlist(prompt: str, bundle: dict, model_key: str,
                      metadata: dict, n_songs: int = 20):
    """
    Dataset-free playlist pipeline:
      1. NLP → emotion weights
      2. Weighted audio centroid → target audio profile
      3. Generate ~800 synthetic candidates via Gaussian noise around centroid
      4. ML classify all candidates
      5. Filter to dominant emotion(s), rank by confidence → return top-n
    """
    weights    = parse_prompt_to_weights(prompt, bundle)
    dominant   = max(weights, key=weights.get)
    centroids  = get_centroids(metadata)

    # Target audio = weighted blend of per-emotion centroids
    target = {
        col: sum(weights[em] * centroids.get(em, {}).get(col, AUDIO_RANGES[col][2])
                 for em in weights)
        for col in PRIMARY_AUDIO_COLS
    }

    # Generate synthetic candidate pool
    n_pool = max(800, n_songs * 40)
    rng    = np.random.default_rng(42)
    pool   = []
    for _ in range(n_pool):
        song = {}
        for col, (lo, hi, _) in AUDIO_RANGES.items():
            center = target.get(col, (lo + hi) / 2)
            val    = center + rng.normal(0, 0.15 * (hi - lo))
            song[col] = float(np.clip(val, lo, hi))
        pool.append(song)

    # ML inference on pool
    try:
        X_pool            = build_feature_matrix(pool, [""] * n_pool, bundle)
        pred_labels, proba = run_inference(X_pool, bundle, model_key)
    except Exception as exc:
        st.error(f"Playlist inference error: {exc}")
        return pd.DataFrame(), weights, dominant

    le      = bundle.get("label_encoder")
    classes = list(le.classes_) if le is not None else list(EMOTION_STYLE.keys())

    pool_df = pd.DataFrame(pool)
    pool_df["_pred_em"]    = pred_labels
    pool_df["_confidence"] = proba.max(axis=1)
    for i, em in enumerate(classes):
        pool_df[f"_conf_{em}"] = proba[:, i]

    # Filter to top-2 emotion classes
    top2       = sorted(weights, key=weights.get, reverse=True)[:2]
    candidates = pool_df[pool_df["_pred_em"].isin(top2)].copy()
    if len(candidates) < n_songs:
        candidates = pool_df[pool_df["_pred_em"] == dominant].copy()
    if len(candidates) < n_songs:
        candidates = pool_df.copy()

    # Rank by model confidence in dominant emotion
    conf_col = f"_conf_{dominant}"
    rank_col = conf_col if conf_col in candidates.columns else "_confidence"
    playlist = candidates.nlargest(n_songs, rank_col)
    return playlist.reset_index(drop=True), weights, dominant


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
bundle   = load_models()
metadata = load_metadata()
perf     = metadata.get("model_performance", {})

AVAILABLE_MODELS = [n for n in MODEL_FILES if bundle and n in bundle]

if not bundle or not AVAILABLE_MODELS:
    st.error(
        "❌ No classifier models found in `model/`.\n\n"
        "Make sure `model/lightgbm_model.pkl`, `model/tfidf.pkl`, etc. exist."
    )
    st.stop()

best_acc = max((v.get("accuracy",    0) for v in perf.values()), default=0)
best_f1  = max((v.get("f1_weighted", 0) for v in perf.values()), default=0)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎵 MoodSense")
    st.markdown("**Music Emotion Classifier**")
    st.markdown("---")

    st.markdown("#### 🤖 Active Model")
    def _lbl(n: str) -> str:
        p   = perf.get(n.lower(), {})
        acc = p.get("accuracy")
        return f"{n} ({acc*100:.1f}%)" if acc else n

    model_choice_label = st.selectbox("Model", [_lbl(m) for m in AVAILABLE_MODELS])
    model_key          = model_choice_label.split(" ")[0]

    st.markdown("---")
    st.markdown("#### 📊 Model Stats")
    if best_acc: st.metric("Best Test Accuracy", f"{best_acc*100:.2f}%")
    if best_f1:  st.metric("Best Weighted F1",   f"{best_f1:.4f}")
    n_feat = metadata.get("num_features")
    if n_feat: st.metric("Feature Dimensions",   f"{n_feat:,}")

    st.markdown("---")
    st.markdown("#### 📚 Project")
    st.markdown("**Course:** MS DSP 422\n\n**Team:** Group 3")
    if metadata.get("training_date"):
        st.caption(f"Trained: {metadata['training_date']}")

    st.markdown("---")
    st.markdown("#### 🎭 Emotions")
    for em, s in EMOTION_STYLE.items():
        st.markdown(f"{s['emoji']} **{em.title()}**")


# ══════════════════════════════════════════════════════════════════════════════
# HERO + KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1 class="hero-title">🎵 MoodSense</h1>
  <p class="hero-sub">AI-Powered Music Emotion Classification · MS DSP 422 Group 3</p>
</div>
""", unsafe_allow_html=True)

n_feat_v  = metadata.get("num_features", "—")
n_train_v = metadata.get("training_samples", "—")

for col, val, lbl in zip(
    st.columns(4),
    [f"{best_acc*100:.1f}%" if best_acc else "—",
     f"{best_f1:.4f}"       if best_f1  else "—",
     f"{n_feat_v:,}" if isinstance(n_feat_v, int) else str(n_feat_v),
     f"{n_train_v:,}" if isinstance(n_train_v, int) else str(n_train_v)],
    ["Best Accuracy", "Best Weighted F1", "Feature Dimensions", "Training Samples"],
):
    col.markdown(
        f'<div class="kpi"><p class="kpi-val">{val}</p>'
        f'<p class="kpi-lbl">{lbl}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Classify Song",
    "📋 Batch Classify",
    "🗺️ Emotion Atlas",
    "🔬 Model Performance",
    "🎧 Your Mood, Your Playlist",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLASSIFY A SINGLE SONG
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🎯 Classify a Song's Emotion")
    st.caption("Enter audio features and optional lyrics → the ML model predicts emotion.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Song Info")
        song_name   = st.text_input("Song Title",  placeholder="e.g. Bohemian Rhapsody")
        artist_name = st.text_input("Artist",       placeholder="e.g. Queen")
        lyrics      = st.text_area(
            "Lyrics (optional — activates TF-IDF features)",
            placeholder="Paste lyrics here…", height=180,
        )

    with col_b:
        st.subheader("Audio Features")
        energy   = st.slider("⚡ Energy",           0.0, 1.0, 0.65, 0.01, key="s1_e")
        dance    = st.slider("💃 Danceability",     0.0, 1.0, 0.60, 0.01, key="s1_d")
        pos      = st.slider("😊 Positiveness",     0.0, 1.0, 0.50, 0.01, key="s1_p")
        acoustic = st.slider("🎸 Acousticness",     0.0, 1.0, 0.30, 0.01, key="s1_ac")
        tempo    = st.slider("🥁 Tempo (BPM)",      60,  200, 120,  1,    key="s1_t")
        speech   = st.slider("🗣 Speechiness",      0.0, 1.0, 0.05, 0.01, key="s1_sp")
        instr    = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.10, 0.01, key="s1_i")
        loudness = st.slider("🔊 Loudness (dB)",   -60.0, 0.0, -7.0, 0.5, key="s1_l")
        pop      = st.slider("⭐ Popularity",       0,   100, 50,    1,   key="s1_pp")

    st.markdown("---")
    if st.button("🎵 Classify Emotion", type="primary", use_container_width=True, key="btn_classify"):
        song_vals = {
            "Energy": energy, "Danceability": dance, "Positiveness": pos,
            "Acousticness": acoustic, "Tempo": tempo, "Speechiness": speech,
            "Instrumentalness": instr, "Loudness": loudness,
            "Popularity": pop, "Liveness": 0.1,
        }
        with st.spinner("Running ML inference…"):
            X          = build_feature_matrix([song_vals], [lyrics], bundle)
            labels, pr = run_inference(X, bundle, model_key)

        em     = str(labels[0])
        pr0    = pr[0]
        le_    = bundle.get("label_encoder")
        cls    = list(le_.classes_) if le_ is not None else list(EMOTION_STYLE.keys())
        conf   = float(pr0.max())
        style  = EMOTION_STYLE.get(em, {"color":"#64748b","emoji":"🎵","desc":""})
        probs  = {str(c): float(p) for c, p in zip(cls, pr0)}

        st.success("✅ Classification complete!")
        r1, r2 = st.columns(2)

        with r1:
            st.markdown(
                f'<div class="emo-card" style="background:{style["color"]};color:white;">'
                f'<div class="emo-emoji">{style["emoji"]}</div>'
                f'<div class="emo-name">{em.upper()}</div>'
                f'<div class="emo-desc">{style["desc"]}</div>'
                f'</div>', unsafe_allow_html=True,
            )
            level = "High" if conf > .65 else "Medium" if conf > .45 else "Low"
            st.metric("Model Confidence", f"{conf*100:.1f}%", delta=level)
            if song_name:
                st.markdown(f"**🎵 {song_name}**" +
                            (f" — *{artist_name}*" if artist_name else ""))
            st.caption(f"Model: {model_key}")

        with r2:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                "Emotion":     [e.title() for e in probs],
                "Probability": list(probs.values()),
            }).sort_values("Probability", ascending=False)
            fig = px.bar(
                prob_df, x="Emotion", y="Probability", color="Emotion",
                color_discrete_map={e.title(): EMOTION_STYLE.get(e, {}).get("color","#ccc")
                                    for e in probs},
                text="Probability",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(showlegend=False, height=320, yaxis_range=[0,1],
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📋 Batch Classify")
    st.caption("Classify multiple songs at once via CSV upload or manual entry. "
               "The ML model runs on every row.")

    # Template download
    tmpl_cols = ["song", "artist", "lyrics"] + PRIMARY_AUDIO_COLS
    tmpl_row  = {"song":"Example Song","artist":"Example Artist","lyrics":"",
                 "Energy":0.75,"Danceability":0.72,"Positiveness":0.78,
                 "Speechiness":0.07,"Liveness":0.17,"Acousticness":0.22,
                 "Instrumentalness":0.04,"Tempo":128,"Loudness":-5,"Popularity":65}
    st.download_button(
        "📥 Download CSV Template",
        pd.DataFrame([tmpl_row])[tmpl_cols].to_csv(index=False).encode(),
        "moodsense_batch_template.csv", "text/csv",
    )

    batch_mode = st.radio("Input method:", ["Upload CSV", "Manual entry"], horizontal=True)
    songs_to_classify: list[dict] = []

    if batch_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV", type="csv")
        if uploaded:
            try:
                up_df = pd.read_csv(uploaded)
                st.info(f"Loaded {len(up_df)} rows.")
                st.dataframe(up_df.head(5), use_container_width=True)
                for _, row in up_df.iterrows():
                    songs_to_classify.append({
                        "_song":   str(row.get("song",   "—")),
                        "_artist": str(row.get("artist", "—")),
                        "_lyrics": str(row.get("lyrics", "")),
                        **{col: float(row[col]) if col in row.index and pd.notna(row[col])
                                else AUDIO_RANGES[col][2]
                           for col in PRIMARY_AUDIO_COLS},
                    })
            except Exception as exc:
                st.error(f"CSV parse error: {exc}")
    else:
        n_manual = st.number_input("Number of songs", 1, 20, 3, 1)
        for i in range(int(n_manual)):
            with st.expander(f"Song #{i+1}", expanded=(i == 0)):
                mc1, mc2 = st.columns(2)
                with mc1:
                    sn = st.text_input("Title",  key=f"mn_{i}", placeholder="Song title")
                    sa = st.text_input("Artist", key=f"ma_{i}", placeholder="Artist name")
                    sl = st.text_area("Lyrics (optional)", key=f"ml_{i}", height=80)
                with mc2:
                    se  = st.slider("Energy",       0.0,1.0,0.65,0.01, key=f"me_{i}")
                    sd  = st.slider("Danceability", 0.0,1.0,0.60,0.01, key=f"md_{i}")
                    sp_ = st.slider("Positiveness", 0.0,1.0,0.50,0.01, key=f"mp_{i}")
                    st_ = st.slider("Tempo (BPM)",  60,200,120,1,       key=f"mt_{i}")
                    sac = st.slider("Acousticness", 0.0,1.0,0.30,0.01, key=f"mac_{i}")
                songs_to_classify.append({
                    "_song":sn or f"Song {i+1}","_artist":sa or "Unknown","_lyrics":sl,
                    "Energy":se,"Danceability":sd,"Positiveness":sp_,"Tempo":float(st_),
                    "Acousticness":sac,"Speechiness":0.05,"Liveness":0.1,
                    "Instrumentalness":0.1,"Loudness":-7.0,"Popularity":50,
                })

    if st.button("🎵 Classify All Songs", type="primary",
                 use_container_width=True, key="btn_batch",
                 disabled=len(songs_to_classify) == 0):
        audio_rows  = [{k:v for k,v in s.items() if not k.startswith("_")}
                       for s in songs_to_classify]
        lyrics_list = [s.get("_lyrics","") for s in songs_to_classify]

        with st.spinner(f"Running ML on {len(audio_rows)} songs…"):
            X_b            = build_feature_matrix(audio_rows, lyrics_list, bundle)
            b_labels, b_pr = run_inference(X_b, bundle, model_key)

        le_     = bundle.get("label_encoder")
        cls     = list(le_.classes_) if le_ is not None else list(EMOTION_STYLE.keys())
        st.success(f"✅ Classified {len(audio_rows)} songs!")
        st.markdown("---")
        st.markdown(f"### Results — {model_key}")

        result_rows = []
        for idx, (song, lbl, pr_row) in enumerate(
                zip(songs_to_classify, b_labels, b_pr), 1):
            em       = str(lbl)
            conf     = float(pr_row.max())
            es       = EMOTION_STYLE.get(em, {"color":"#64748b","emoji":"🎵"})
            conf_pct = int(conf * 100)
            st.markdown(
                f'<div class="batch-row">'
                f'<span class="batch-num">{idx}</span>'
                f'<div class="batch-info">'
                f'<div class="batch-name">{song.get("_song","—")}</div>'
                f'<div class="batch-artist">{song.get("_artist","—")}</div>'
                f'</div>'
                f'<span class="badge" style="background:{es["color"]}22;color:{es["color"]};">'
                f'{es["emoji"]} {em.title()}</span>'
                f'<div class="conf-wrap" title="{conf_pct}% confidence">'
                f'<div class="conf-bg"><div class="conf-fg" '
                f'style="width:{conf_pct}%;background:{es["color"]};"></div></div>'
                f'</div></div>', unsafe_allow_html=True,
            )
            result_rows.append({
                "Song":             song.get("_song","—"),
                "Artist":           song.get("_artist","—"),
                "Predicted Emotion":em,
                "Confidence":       round(conf, 4),
                **{f"P({c})": round(float(p),4) for c,p in zip(cls, pr_row)},
            })

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            f"📥 Download Results CSV ({len(result_rows)} songs)",
            pd.DataFrame(result_rows).to_csv(index=False).encode(),
            "moodsense_batch_results.csv", "text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Batch Summary")
        lc = pd.Series(b_labels).value_counts()
        fig_pie = px.pie(
            values=lc.values, names=[e.title() for e in lc.index], color=lc.index,
            color_discrete_map={e: EMOTION_STYLE.get(e,{}).get("color","#ccc") for e in lc.index},
            hole=0.4, title="Emotion Distribution in Batch",
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EMOTION ATLAS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🗺️ Emotion Atlas")
    st.caption("Explore what each emotion sounds like in audio feature space. "
               "Adjust sliders to see the model predict in real time.")

    centroids = get_centroids(metadata)

    # ── Emotion profile cards ─────────────────────────────────────────────
    st.subheader("Audio Fingerprint per Emotion")
    feat_labels = {
        "Energy":"⚡","Danceability":"💃","Positiveness":"😊",
        "Acousticness":"🎸","Speechiness":"🗣","Tempo":"🥁",
    }
    atlas_cols = st.columns(4)
    for (em, s), col in zip(EMOTION_STYLE.items(), atlas_cols):
        c = centroids.get(em, {})
        with col:
            st.markdown(
                f'<div class="atlas-card" style="border-top:4px solid {s["color"]};">'
                f'<div class="atlas-title">{s["emoji"]} {em.title()}</div>'
                f'<div class="atlas-sub">{s["desc"]}</div>',
                unsafe_allow_html=True,
            )
            for feat, icon in feat_labels.items():
                raw     = c.get(feat, AUDIO_RANGES[feat][2])
                lo, hi, _ = AUDIO_RANGES[feat]
                pct     = int((raw - lo) / (hi - lo) * 100)
                label   = f"{raw:.0f}" if feat == "Tempo" else f"{raw:.2f}"
                st.markdown(
                    f'<div style="margin:.35rem 0;">'
                    f'<span style="font-size:.78rem;color:#64748b;">{icon} {feat}</span>'
                    f'<div style="background:#e2e8f0;border-radius:4px;height:6px;margin:.2rem 0;">'
                    f'<div style="width:{pct}%;background:{s["color"]};border-radius:4px;height:6px;"></div>'
                    f'</div>'
                    f'<span style="font-size:.75rem;font-weight:600;">{label}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Radar chart ───────────────────────────────────────────────────────
    st.subheader("Comparative Radar Chart")
    radar_feats = ["Energy","Danceability","Positiveness","Acousticness","Speechiness"]
    fig_radar   = go.Figure()
    for em, s in EMOTION_STYLE.items():
        c    = centroids.get(em, {})
        vals = [c.get(f, 0.5) for f in radar_feats] + [c.get(radar_feats[0], 0.5)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_feats + [radar_feats[0]],
            fill="toself", name=em.title(),
            line=dict(color=s["color"]), fillcolor=s["color"], opacity=0.3,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, height=440,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ── Live prediction explorer ──────────────────────────────────────────
    st.subheader("🎛️ Live Prediction Explorer")
    st.caption("Adjust sliders — the model predicts on every change, no button required.")

    lp1, lp2 = st.columns([1, 1])
    with lp1:
        lp_e  = st.slider("⚡ Energy",           0.0,1.0,0.65,0.01, key="lp_e")
        lp_d  = st.slider("💃 Danceability",     0.0,1.0,0.60,0.01, key="lp_d")
        lp_p  = st.slider("😊 Positiveness",     0.0,1.0,0.50,0.01, key="lp_p")
        lp_ac = st.slider("🎸 Acousticness",     0.0,1.0,0.30,0.01, key="lp_ac")
        lp_t  = st.slider("🥁 Tempo (BPM)",      60,200,120,1,       key="lp_t")

    lp_vals = {
        "Energy":lp_e,"Danceability":lp_d,"Positiveness":lp_p,
        "Acousticness":lp_ac,"Tempo":float(lp_t),
        "Speechiness":0.05,"Liveness":0.1,"Instrumentalness":0.1,
        "Loudness":-7.0,"Popularity":50,
    }
    try:
        X_lp           = build_feature_matrix([lp_vals], [""], bundle)
        lp_lbl, lp_pr  = run_inference(X_lp, bundle, model_key)
        lp_em    = str(lp_lbl[0])
        lp_style = EMOTION_STYLE.get(lp_em, {"color":"#64748b","emoji":"🎵","desc":""})
        lp_conf  = float(lp_pr[0].max())
        le_      = bundle.get("label_encoder")
        lp_cls   = list(le_.classes_) if le_ is not None else list(EMOTION_STYLE.keys())
        lp_probs = {str(c): float(p) for c, p in zip(lp_cls, lp_pr[0])}

        with lp2:
            st.markdown(
                f'<div class="emo-card" style="background:{lp_style["color"]};color:white;padding:1.5rem;">'
                f'<div style="font-size:3.5rem;">{lp_style["emoji"]}</div>'
                f'<div class="emo-name" style="font-size:1.8rem;">{lp_em.upper()}</div>'
                f'<div class="emo-desc">{lp_conf*100:.1f}% confidence</div>'
                f'</div>', unsafe_allow_html=True,
            )
            df_lp = pd.DataFrame({
                "Emotion":     [e.title() for e in lp_probs],
                "Probability": list(lp_probs.values()),
            }).sort_values("Probability", ascending=False)
            fig_lp = px.bar(
                df_lp, x="Emotion", y="Probability", color="Emotion",
                color_discrete_map={e.title(): EMOTION_STYLE.get(e,{}).get("color","#ccc")
                                    for e in lp_probs},
                text="Probability",
            )
            fig_lp.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_lp.update_layout(showlegend=False, height=260, yaxis_range=[0,1],
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                 margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig_lp, use_container_width=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🔬 Model Performance")

    if not perf:
        st.warning("No model_metadata.json found — performance data unavailable.")
    else:
        best_name = max(perf, key=lambda m: perf[m].get("accuracy", 0))
        best_m    = perf[best_name]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best Model",       best_name)
        k2.metric("Test Accuracy",    f"{best_m.get('accuracy',0)*100:.2f}%")
        k3.metric("Weighted F1",      f"{best_m.get('f1_weighted',0):.4f}")
        n_tr = metadata.get("training_samples","—")
        k4.metric("Training Samples", f"{n_tr:,}" if isinstance(n_tr,int) else str(n_tr))

        st.markdown("---")
        st.subheader("📊 Model Comparison")
        perf_df = pd.DataFrame([
            {"Model":m,"Accuracy":v.get("accuracy",0),"Weighted F1":v.get("f1_weighted",0)}
            for m,v in perf.items()
        ]).sort_values("Accuracy", ascending=False)
        fig_perf = px.bar(
            perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="All Models — Accuracy & Weighted F1",
            color_discrete_sequence=["#1d4ed8","#3b82f6"],
        )
        fig_perf.update_layout(height=420, yaxis_range=[0,1])
        st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Breakdown")
        fb = metadata.get("feature_breakdown", {})
        if fb:
            c1f,c2f,c3f = st.columns(3)
            c1f.metric("TF-IDF Features", f"{fb.get('tfidf',0):,}")
            c2f.metric("Audio Features",  f"{fb.get('audio',0):,}")
            c3f.metric("VADER Features",  f"{fb.get('vader',0):,}")
            fb_fig = px.bar(
                pd.DataFrame({"Feature Group":list(fb.keys()),"Count":list(fb.values())}),
                x="Feature Group", y="Count", color="Feature Group",
                color_discrete_sequence=["#0f172a","#1d4ed8","#3b82f6"],
                title="Features by Type",
            )
            fb_fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fb_fig, use_container_width=True)
        else:
            st.info("Feature breakdown not found in metadata.")

        st.markdown("---")
        st.subheader("📋 Full Results Table")
        raw_tbl = pd.DataFrame([
            {"Model":m,"Accuracy":f"{v.get('accuracy',0)*100:.2f}%",
             "Weighted F1":f"{v.get('f1_weighted',0):.4f}"}
            for m,v in perf.items()
        ]).sort_values("Accuracy", ascending=False)
        st.dataframe(raw_tbl, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — YOUR MOOD, YOUR PLAYLIST
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 60%,#7c3aed 100%);
                padding:2.5rem 2rem;border-radius:18px;color:white;margin-bottom:2rem;">
        <h2 style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;margin:0 0 .5rem;">
            🎧 Your Mood, Your Playlist
        </h2>
        <p style="opacity:.85;margin:0;font-size:1rem;">
            Describe a vibe, a moment, a feeling. NLP maps your words to emotion space,
            derives a target audio profile, generates synthetic candidates via Gaussian noise,
            then runs the ML classifier to curate your playlist. No dataset. All model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_pre, col_custom = st.columns([1, 2], gap="large")

    with col_pre:
        st.markdown("#### 🎛️ Pre-designed Vibes")
        st.caption("Click any card to instantly generate a playlist")
        for p in PRESET_PLAYLISTS:
            if st.button(
                f"{p['icon']}  {p['title']}\n{p['subtitle']}",
                key=f"preset_{p['id']}",
                use_container_width=True,
            ):
                st.session_state.update({
                    "pl_prompt":p["prompt"], "pl_n":p["n"],
                    "pl_title":f"{p['icon']} {p['title']}", "pl_run":True,
                })

    with col_custom:
        st.markdown("#### ✍️ Describe Your Own Vibe")
        st.caption("Type anything — a feeling, a scene, a moment. The model does the rest.")
        prompt_input = st.text_area(
            "Your mood prompt",
            placeholder='e.g. "Sunday morning coffee, soft rain on the window, feeling nostalgic…"',
            height=110, label_visibility="collapsed", key="pl_prompt_input",
        )
        ca, cb, cc = st.columns([2,1,1])
        with ca:
            n_songs_custom = st.slider("Playlist length", 5, 40, 20, 5, key="pl_n_slider")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎵 Generate", type="primary",
                         use_container_width=True, key="pl_gen_btn"):
                if prompt_input.strip():
                    st.session_state.update({
                        "pl_prompt":prompt_input.strip(), "pl_n":n_songs_custom,
                        "pl_title":"🎧 Custom Playlist", "pl_run":True,
                    })
                else:
                    st.warning("Please describe a mood or vibe first.")
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear", use_container_width=True, key="pl_clear_btn"):
                for k in ["pl_prompt","pl_n","pl_title","pl_run","pl_result"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Run generator
    if st.session_state.get("pl_run"):
        st.session_state["pl_run"] = False
        active_prompt = st.session_state.get("pl_prompt", "")
        active_n      = st.session_state.get("pl_n", 20)
        active_title  = st.session_state.get("pl_title", "Your Playlist")
        st.markdown("---")
        with st.spinner("🔮 Generating & classifying synthetic candidates…"):
            pl_df, emo_weights, dominant = generate_playlist(
                prompt=active_prompt, bundle=bundle,
                model_key=model_key, metadata=metadata, n_songs=active_n,
            )
        st.session_state["pl_result"] = {
            "df":pl_df, "weights":emo_weights, "dominant":dominant,
            "title":active_title, "prompt":active_prompt,
        }

    # Render result
    res = st.session_state.get("pl_result")
    if res and not res["df"].empty:
        pl_df   = res["df"];  weights  = res["weights"]
        dominant= res["dominant"]; pl_title = res["title"]; pl_prompt = res["prompt"]
        dom_s   = EMOTION_STYLE.get(dominant, {"color":"#1d4ed8","emoji":"🎵"})

        st.markdown("---")
        st.markdown(
            f'<div class="playlist-card" style="border-top:4px solid {dom_s["color"]};">'
            f'<div class="playlist-header">{pl_title}</div>'
            f'<div class="playlist-meta">Prompt: <em>"{pl_prompt}"</em> &nbsp;·&nbsp; '
            f'{len(pl_df)} tracks &nbsp;·&nbsp; '
            f'Dominant: <strong>{dominant.title()}</strong> {dom_s["emoji"]}</div>'
            f'</div>', unsafe_allow_html=True,
        )

        rr1, rr2 = st.columns([1,2])
        with rr1:
            st.markdown("**NLP Emotion Signal**")
            st.caption("VADER + TF-IDF cosine → emotion space")
            w_df = pd.DataFrame({
                "Emotion":[e.title() for e in weights],
                "Weight":list(weights.values()),
            }).sort_values("Weight")
            fig_w = px.bar(
                w_df, x="Weight", y="Emotion", orientation="h", color="Emotion",
                color_discrete_map={e.title(): EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                text="Weight",
            )
            fig_w.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_w.update_layout(showlegend=False, height=220,
                                xaxis_range=[0, max(weights.values())*1.35],
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0,r=40,t=10,b=10))
            st.plotly_chart(fig_w, use_container_width=True)

        with rr2:
            st.markdown("**Playlist Audio Profile**")
            st.caption("Each point = a model-classified synthetic candidate")
            if "Energy" in pl_df.columns and "Positiveness" in pl_df.columns:
                sc = px.scatter(
                    pl_df, x="Energy", y="Positiveness", color="_pred_em",
                    color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                    opacity=.8, height=220,
                )
                sc.update_layout(showlegend=True, margin=dict(l=0,r=0,t=10,b=10),
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(sc, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### 🎵 Playlist — {len(pl_df)} AI-Curated Tracks")
        st.caption("Synthetic audio profiles generated and classified by the ML model.")

        conf_col = (f"_conf_{dominant}" if f"_conf_{dominant}" in pl_df.columns
                    else "_confidence")

        for idx, (_, row) in enumerate(pl_df.iterrows(), 1):
            em_r   = str(row.get("_pred_em", dominant))
            es     = EMOTION_STYLE.get(em_r, {"color":"#64748b","emoji":"🎵"})
            c_val  = float(row[conf_col]) if conf_col in pl_df.columns and pd.notna(row.get(conf_col)) else 0.0
            c_pct  = int(c_val * 100)
            e_v    = float(row.get("Energy",      0))
            d_v    = float(row.get("Danceability", 0))
            p_v    = float(row.get("Positiveness", 0))
            t_v    = float(row.get("Tempo",       120))
            st.markdown(
                f'<div class="track-row">'
                f'<span class="track-num">{idx}</span>'
                f'<div class="track-info">'
                f'<div class="track-name">Track {idx:02d} &nbsp;·&nbsp; '
                f'⚡{e_v:.2f} 💃{d_v:.2f} 😊{p_v:.2f} 🥁{t_v:.0f}BPM</div>'
                f'<div class="track-artist">AI-generated audio profile · {c_pct}% confidence</div>'
                f'</div>'
                f'<span class="badge" style="background:{es["color"]}22;color:{es["color"]};">'
                f'{es["emoji"]} {em_r.title()}</span>'
                f'<div class="conf-wrap"><div class="conf-bg">'
                f'<div class="conf-fg" style="width:{c_pct}%;background:{es["color"]};"></div>'
                f'</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        export_cols = [c for c in ["Energy","Danceability","Positiveness","Speechiness",
                                    "Acousticness","Tempo","Loudness","_pred_em","_confidence"]
                       if c in pl_df.columns]
        st.download_button(
            f"📥 Export Audio Profiles CSV ({len(pl_df)} tracks)",
            pl_df[export_cols].rename(columns={
                "_pred_em":"Predicted Emotion","_confidence":"Model Confidence",
            }).to_csv(index=False).encode(),
            "moodsense_playlist.csv", "text/csv",
            use_container_width=True,
        )

    elif st.session_state.get("pl_result") and res["df"].empty:
        st.warning("No candidates matched. Try a different prompt or check models are loaded.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748b;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎶</div>
            <p style="font-size:1.1rem;font-weight:600;">Pick a preset or type your own mood above.</p>
            <p style="font-size:.9rem;">
                The ML model generates synthetic audio profiles matching your emotional signal.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:2rem;
            background:linear-gradient(135deg,#0f172a,#1e293b);
            border-radius:14px;color:#e2e8f0;">
    <p style="font-size:1.3rem;font-weight:800;margin:0;font-family:Syne,sans-serif;">
        🎵 MoodSense
    </p>
    <p style="opacity:.7;margin:.8rem 0 0;">
        MS DSP 422 – Practical Machine Learning | Group 3<br>
        Ankit Mittal · Albin Anto Jose · Nandini Bag · Kasheena Mulla
    </p>
</div>
""", unsafe_allow_html=True)
