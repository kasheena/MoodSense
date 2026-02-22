"""
MoodSense: Music Emotion Classification Platform
─────────────────────────────────────────────────
MS DSP 422 - Practical Machine Learning | Group 3

GitHub repo layout expected:
    moodsense-app/
    ├── moodsense_app.py        ← this file
    ├── requirements.txt
    ├── model/
    │   ├── audio_scaler.pkl
    │   ├── vader_scaler.pkl    (optional)
    │   ├── tfidf.pkl
    │   ├── label_encoder.pkl
    │   ├── lightgbm_model.pkl
    │   ├── xgboost_model.pkl
    │   ├── ensemble_model.pkl
    │   ├── linearsvc_model.pkl
    │   ├── logistic_model.pkl
    │   └── model_metadata.json
    └── data/
        └── spotify_dataset_withemotion.csv   ← place dataset here

Run locally:
    streamlit run moodsense_app.py

Run on Streamlit Cloud:
    Push repo to GitHub → deploy via share.streamlit.io
    Upload large dataset via st.file_uploader or host on Google Drive / HuggingFace Hub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import re
import json
import warnings
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED ENSEMBLE CLASS
# Must be defined before ANY pickle.load call so unpickling works correctly.
# ══════════════════════════════════════════════════════════════════════════════
class WeightedEnsemble:
    """Weighted soft-voting ensemble of LightGBM, XGBoost, and Logistic Regression."""

    def __init__(self, models: dict, weights: dict, label_encoder):
        self.models = models
        self.weights = weights
        self.label_encoder = label_encoder

    def predict_proba(self, X):
        probas = []
        total = sum(self.weights.values())
        for name, model in self.models.items():
            w = self.weights[name] / total
            probas.append(model.predict_proba(X) * w)
        return np.sum(probas, axis=0)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder.inverse_transform(idx)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS — all relative to THIS file's directory (repo root)
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = Path(__file__).parent.resolve()

MODEL_DIR    = _ROOT / "model"
DATA_DIR     = _ROOT / "data"
DATASET_PATH = DATA_DIR / "spotify_dataset_withemotion.csv"

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

# Same label mapping as training pipeline
EMOTION_4_MAP = {
    "joy": "happy", "surprise": "happy",
    "sadness": "sad", "fear": "sad",
    "anger": "anger", "angry": "anger",
    "Love": "love",  "love": "love",
}

# Audio feature columns — must match training order exactly
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
ENGINEERED_COLS = [
    "feel_good", "electronic", "intensity", "vocal_dom",
    "mood_energy", "valence_dance", "arousal",
]

# Pre-designed playlist presets (prompts fed to the ML pipeline — no special logic)
PRESET_PLAYLISTS = [
    {"id": "work_focus",      "icon": "💼", "title": "Deep Work Focus",
     "subtitle": "Instrumental · low distraction · steady tempo",
     "prompt": "instrumental focus concentration work study calm steady beats no lyrics", "n": 20},
    {"id": "rainy_afternoon", "icon": "🌧️", "title": "Rainy Afternoon",
     "subtitle": "Melancholic · acoustic · introspective",
     "prompt": "rain melancholic quiet acoustic reflective slow afternoon sad gentle", "n": 18},
    {"id": "heartbreak",      "icon": "💔", "title": "Heartbreak Hotel",
     "subtitle": "Emotional · raw · post-breakup vibes",
     "prompt": "heartbreak crying pain love lost breakup emotional tears missing someone", "n": 18},
    {"id": "late_night_drive","icon": "🚗", "title": "Late Night Drive",
     "subtitle": "Dark · atmospheric · driving energy",
     "prompt": "night dark driving highway electric atmospheric moody intensity beat", "n": 20},
    {"id": "summer_party",    "icon": "🌞", "title": "Summer Party",
     "subtitle": "High energy · danceable · joyful",
     "prompt": "party dance happy upbeat energy summer fun celebration joyful loud", "n": 20},
    {"id": "romance",         "icon": "🌹", "title": "Date Night",
     "subtitle": "Romantic · tender · warm",
     "prompt": "love romance tender sweet gentle warmth affection intimate caring", "n": 18},
    {"id": "anger_release",   "icon": "⚡", "title": "Rage Release",
     "subtitle": "Intense · powerful · cathartic",
     "prompt": "anger intense powerful aggressive rage frustration electric loud cathartic", "n": 18},
    {"id": "morning_run",     "icon": "🏃", "title": "Morning Run",
     "subtitle": "Energetic · motivating · fast tempo",
     "prompt": "run exercise morning energetic motivating fast beats workout pump adrenaline", "n": 20},
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
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
    --ink:    #0f172a;
    --muted:  #64748b;
    --border: #e2e8f0;
    --bg:     #f8fafc;
    --card:   #ffffff;
    --blue:   #1d4ed8;
    --blue-lt:#3b82f6;
}

.main { background: var(--bg); }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #3b82f6 100%);
    padding: 3.5rem 2.5rem; border-radius: 20px; color: white;
    text-align: center; margin-bottom: 2rem;
    box-shadow: 0 24px 64px rgba(29,78,216,.25);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-title { font-size: 3.5rem; font-weight: 800; margin: 0 0 1rem; letter-spacing: -0.03em; }
.hero-sub   { font-size: 1.2rem; opacity: .85; margin: 0; }

/* ── KPI cards ── */
.kpi {
    background: var(--card); padding: 1.5rem; border-radius: 14px;
    border: 1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.kpi-val { font-size: 2.2rem; font-weight: 800; color: var(--blue);
           margin: 0; font-family: 'Syne', sans-serif; }
.kpi-lbl { font-size: .75rem; color: var(--muted); text-transform: uppercase;
           letter-spacing: .08em; font-weight: 600; margin-top: .4rem; }

/* ── Emotion result card ── */
.emo-card  { padding: 2.5rem; border-radius: 18px; text-align: center;
             box-shadow: 0 12px 40px rgba(0,0,0,.18); margin: 1.5rem 0; }
.emo-emoji { font-size: 5rem; }
.emo-name  { font-size: 2.2rem; font-weight: 800; margin: .8rem 0 .4rem;
             font-family: 'Syne', sans-serif; }
.emo-desc  { font-size: 1rem; opacity: .85; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a, #1e293b); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, var(--blue), var(--blue-lt));
    color: white; border: none; border-radius: 10px;
    padding: .75rem 2rem; font-weight: 600; letter-spacing: .02em;
    transition: all .25s;
}
.stButton button:hover {
    box-shadow: 0 8px 24px rgba(29,78,216,.35);
    transform: translateY(-2px);
}

/* ── Playlist tab ── */
.playlist-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.75rem;
    box-shadow: 0 4px 16px rgba(0,0,0,.06); margin-bottom: 1rem;
}
.playlist-header {
    font-family: 'Syne', sans-serif; font-size: 1.6rem;
    font-weight: 800; margin: 0 0 .3rem;
}
.playlist-meta { color: var(--muted); font-size: .85rem; margin-bottom: 1rem; }

.track-row {
    display: flex; align-items: center; gap: 1rem;
    padding: .65rem .5rem; border-radius: 8px;
    border-bottom: 1px solid var(--border); transition: background .15s;
}
.track-row:hover { background: var(--bg); }
.track-num   { width: 1.8rem; text-align: right; color: var(--muted);
               font-size: .85rem; font-weight: 600; flex-shrink: 0; }
.track-info  { flex: 1; min-width: 0; }
.track-name  { font-weight: 600; white-space: nowrap;
               overflow: hidden; text-overflow: ellipsis; }
.track-artist{ font-size: .8rem; color: var(--muted); }
.track-badge { font-size: .72rem; font-weight: 700; padding: .2rem .65rem;
               border-radius: 999px; flex-shrink: 0; }
.conf-bar-wrap{ width: 80px; flex-shrink: 0; }
.conf-bar-bg  { background: #e2e8f0; border-radius: 4px; height: 6px; }
.conf-bar-fg  { border-radius: 4px; height: 6px; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS  (cached so they run once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML models…")
def load_models() -> dict | None:
    """Load all pkl artifacts from model/ directory. Returns bundle dict or None."""
    bundle: dict = {}

    def _load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    missing = []
    for required in [AUDIO_SCALER_PATH, TFIDF_PATH, LABEL_ENCODER_PATH]:
        if not required.exists():
            missing.append(str(required))

    if missing:
        st.error(
            f"❌ Required model files not found:\n" + "\n".join(f"  • {m}" for m in missing)
            + "\n\nEnsure `model/` directory contains all pkl files."
        )
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

        loaded_models = [n for n in MODEL_FILES if n in bundle]
        if not loaded_models:
            st.warning("⚠️ No classifier models found in model/ — only preprocessors loaded.")

        return bundle

    except Exception as exc:
        st.error(f"❌ Model loading failed: {exc}")
        return None


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame | None:
    """
    Load the emotion dataset from data/spotify_dataset_withemotion.csv.
    Derives emotion_4 column using the same mapping as training if absent.
    Supports uploading via the sidebar when the file is not on disk.
    """
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)

    # Derive emotion_4 if column absent (raw dataset ships with 'emotion')
    if "emotion_4" not in df.columns:
        if "emotion" not in df.columns:
            st.error("❌ Dataset has neither 'emotion' nor 'emotion_4' column.")
            return None
        df["emotion"] = df["emotion"].replace({"Love": "love", "angry": "anger"})
        df["emotion_4"] = df["emotion"].map(EMOTION_4_MAP)

    valid = list(EMOTION_STYLE.keys())
    df = df[df["emotion_4"].isin(valid)].copy()

    if "Artist(s)" in df.columns and "song" in df.columns:
        df = df.drop_duplicates(subset=["Artist(s)", "song"], keep="first")

    for col in PRIMARY_AUDIO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (mirrors training pipeline exactly)
# ══════════════════════════════════════════════════════════════════════════════

def clean_lyrics(text: str) -> str:
    """Pre-process lyrics text — identical to training step."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)                           # remove [Chorus] etc.
    text = re.sub(r"\b(la|na|oh|ah|uh|yeah|ooh|mm|hey)\b", " ", text)  # ad-libs
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_feature_vector(song_vals: dict, lyrics: str, bundle: dict):
    """
    Construct the exact feature matrix the trained models expect.

    Column layout (25 033 total):
        TF-IDF   : 25 000
        VADER    :      4
        Audio    :     29  (10 primary + 12 API zeros + 7 engineered)
    """
    tfidf = bundle.get("tfidf")

    # 1. TF-IDF
    if tfidf and lyrics.strip():
        X_tfidf = tfidf.transform([clean_lyrics(lyrics)])
    else:
        n = tfidf.get_feature_names_out().shape[0] if tfidf else 25_000
        X_tfidf = csr_matrix((1, n))

    # 2. VADER sentiment
    vader_obj = SentimentIntensityAnalyzer()
    vs = vader_obj.polarity_scores(clean_lyrics(lyrics) if lyrics.strip() else "")
    vader_raw = np.array([[vs["neg"], vs["neu"], vs["pos"], vs["compound"]]])
    if "vader_scaler" in bundle:
        vader_feat = bundle["vader_scaler"].transform(vader_raw).astype("float32")
    else:
        vader_feat = vader_raw.astype("float32")

    # 3. Primary audio
    def sv(k, default=0.0):
        return float(song_vals.get(k, default))

    energy   = sv("Energy",           0.5)
    dance    = sv("Danceability",      0.5)
    pos      = sv("Positiveness",      0.5)
    acoustic = sv("Acousticness",      0.5)
    tempo    = sv("Tempo",           120.0)
    speech   = sv("Speechiness",      0.05)
    instr    = sv("Instrumentalness",  0.1)
    loudness = sv("Loudness",         -7.0)
    liveness = sv("Liveness",          0.1)
    pop      = sv("Popularity",       50.0)

    primary    = [energy, dance, pos, speech, liveness, acoustic, instr, tempo, loudness, pop]
    api_zeros  = [0.0] * len(API_AUDIO_COLS)          # API features not available at inference
    engineered = [
        dance * pos / 100,       # feel_good
        energy - acoustic,       # electronic
        tempo * energy / 100,    # intensity
        speech / (instr + 1),    # vocal_dom
        pos / (energy + 1),      # mood_energy
        0.0,                     # valence_dance  (api_valence unavailable)
        0.0,                     # arousal        (api_energy  unavailable)
    ]

    audio_vec = np.array([primary + api_zeros + engineered], dtype="float32")

    # 4. Scale audio
    scaler = bundle.get("audio_scaler")
    if scaler:
        try:
            audio_vec = scaler.transform(audio_vec)
        except Exception:
            pass  # dimension mismatch fallback — pass raw

    return hstack([X_tfidf, csr_matrix(vader_feat), csr_matrix(audio_vec)])


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(X, bundle: dict, model_name: str) -> dict | None:
    """
    Run inference. Handles both predict_proba models (LightGBM, XGBoost, Ensemble, LogReg)
    and decision-function-only models (LinearSVC).
    Returns: {prediction, confidence, probabilities}
    """
    model = bundle.get(model_name)
    le    = bundle.get("label_encoder")
    if model is None:
        return None

    try:
        if hasattr(model, "predict_proba"):
            proba   = model.predict_proba(X)[0]
            classes = le.classes_ if le is not None else list(EMOTION_STYLE.keys())
            emotion_probs = dict(zip(classes, proba))
            pred_idx      = int(np.argmax(proba))
            prediction    = classes[pred_idx]
        else:
            # LinearSVC — no probability output
            raw        = model.predict(X)[0]
            prediction = str(raw)
            emotion_probs = {e: (1.0 if e == prediction else 0.0) for e in EMOTION_STYLE}
            proba      = np.array(list(emotion_probs.values()))

        # Decode integer label if needed
        if isinstance(prediction, (int, np.integer)) and le is not None:
            prediction = le.inverse_transform([prediction])[0]

        return {
            "prediction":    str(prediction),
            "confidence":    float(np.max(proba)),
            "probabilities": {str(k): float(v) for k, v in emotion_probs.items()},
        }

    except Exception as exc:
        st.error(f"Prediction error ({model_name}): {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR SONGS  (audio cosine distance within predicted emotion)
# ══════════════════════════════════════════════════════════════════════════════

def similar_songs(song_vals: dict, df: pd.DataFrame, emotion: str, n: int = 10) -> pd.DataFrame:
    sub       = df[df["emotion_4"] == emotion].copy()
    feat_cols = [c for c in ["Energy", "Positiveness", "Danceability", "Acousticness", "Tempo"]
                 if c in sub.columns]
    if sub.empty or not feat_cols:
        return pd.DataFrame()

    def dist(row):
        d = 0.0
        for c in feat_cols:
            v1 = 0.0 if pd.isna(row.get(c)) else float(row[c])
            v2 = float(song_vals.get(c, 0.5))
            if c == "Tempo":
                v1 /= 200; v2 /= 200
            d += (v1 - v2) ** 2
        return d ** 0.5

    sub["_dist"] = sub.apply(dist, axis=1)
    return (sub[sub["_dist"] > 0.01]
            .nsmallest(n, "_dist")
            [["song", "Artist(s)", "emotion_4", "Energy", "Positiveness", "_dist"]])


# ══════════════════════════════════════════════════════════════════════════════
# PLAYLIST ENGINE — Pure ML + NLP
# ══════════════════════════════════════════════════════════════════════════════

def parse_prompt_to_signals(prompt_text: str, bundle: dict) -> dict:
    """
    Converts a free-text mood prompt into soft emotion weights.

    Method:
      • VADER polarity scores → continuous 4-d emotion signal (no thresholds)
      • TF-IDF cosine similarity between prompt vector and 4 emotion reference
        documents in the trained vocabulary space
      • 50/50 blend → normalised emotion weight dict
    """
    vader_obj = SentimentIntensityAnalyzer()
    vs        = vader_obj.polarity_scores(prompt_text)
    compound  = vs["compound"]
    neg, pos  = vs["neg"], vs["pos"]

    # Continuous VADER signal — no if/else thresholds
    vader_signal = {
        "happy": pos * (1 + compound) / 2,
        "love":  pos * max(0.0, 1 - neg),
        "sad":   neg * (2 - compound) / 2,
        "anger": neg * (1 + abs(compound)),
    }

    # TF-IDF cosine similarity to emotion reference documents
    EMOTION_DOCS = {
        "happy": "joy dance smile laugh celebrate fun sunshine bright energy cheerful upbeat",
        "sad":   "cry tears sorrow grief lonely pain heartbreak quiet dark rain melancholy",
        "anger": "rage fury fight anger aggressive loud power electric intense frustration scream",
        "love":  "love tender romance sweet kiss warmth affection gentle care heart beloved",
    }
    tfidf_signal = {e: 0.0 for e in EMOTION_STYLE}
    tfidf = bundle.get("tfidf")
    if tfidf:
        try:
            p_vec = tfidf.transform([clean_lyrics(prompt_text)])
            for em, doc in EMOTION_DOCS.items():
                d_vec = tfidf.transform([clean_lyrics(doc)])
                tfidf_signal[em] = float(cosine_similarity(p_vec, d_vec)[0][0])
        except Exception:
            pass

    raw   = {em: 0.5 * vader_signal.get(em, 0.0) + 0.5 * tfidf_signal[em]
             for em in EMOTION_STYLE}
    total = sum(raw.values()) or 1.0
    return {em: v / total for em, v in raw.items()}


@st.cache_data(show_spinner=False)
def get_audio_centroids(_df: pd.DataFrame) -> dict:
    """Mean audio feature vector per emotion from the actual dataset (cached)."""
    audio_cols = [c for c in PRIMARY_AUDIO_COLS if c in _df.columns]
    return {
        em: _df[_df["emotion_4"] == em][audio_cols].mean().to_dict()
        for em in EMOTION_STYLE
    }


def generate_playlist(
    prompt_text: str,
    df: pd.DataFrame,
    bundle: dict,
    model_key: str,
    n_songs: int = 20,
    centroids: dict | None = None,
) -> tuple[pd.DataFrame, dict, str]:
    """
    Playlist generation pipeline — strictly ML/NLP.

    Steps:
      1. NLP → soft emotion weights (parse_prompt_to_signals)
      2. Weighted centroid of per-emotion audio profiles → target audio vector
      3. Sample pool of 5 000 songs, run the trained ML model to predict emotion
         for each (not using BERT ground-truth labels)
      4. Filter to songs predicted in top-2 emotions by weight
      5. Rank by Euclidean distance to target audio vector → return top-n
    """
    if df is None or df.empty or bundle is None:
        return pd.DataFrame(), {}, "happy"

    emotion_weights = parse_prompt_to_signals(prompt_text, bundle)
    dominant_em     = max(emotion_weights, key=emotion_weights.get)

    if centroids is None:
        centroids = get_audio_centroids(df)

    audio_cols   = [c for c in PRIMARY_AUDIO_COLS if c in df.columns]
    target_audio = {
        col: sum(emotion_weights[em] * centroids.get(em, {}).get(col, 0.5)
                 for em in emotion_weights)
        for col in audio_cols
    }

    pool = df.sample(n=min(5_000, len(df)), random_state=42).copy()

    le    = bundle.get("label_encoder")
    model = bundle.get(model_key)

    predicted_emotions = None
    if model is not None and hasattr(model, "predict_proba"):
        try:
            tfidf_obj = bundle.get("tfidf")
            n_tfidf   = tfidf_obj.get_feature_names_out().shape[0] if tfidf_obj else 25_000

            rows = []
            for _, row in pool.iterrows():
                sv  = {c: (float(row[c]) if pd.notna(row.get(c)) else 0.0)
                        for c in PRIMARY_AUDIO_COLS if c in pool.columns}
                e_  = sv.get("Energy",          0.5)
                d_  = sv.get("Danceability",     0.5)
                p_  = sv.get("Positiveness",     0.5)
                a_  = sv.get("Acousticness",     0.5)
                t_  = sv.get("Tempo",          120.0)
                sp_ = sv.get("Speechiness",     0.05)
                i_  = sv.get("Instrumentalness", 0.1)
                primary_v = [sv.get(c, 0.0) for c in PRIMARY_AUDIO_COLS]
                api_v     = [0.0] * len(API_AUDIO_COLS)
                eng_v     = [d_*p_/100, e_-a_, t_*e_/100, sp_/(i_+1), p_/(e_+1), 0.0, 0.0]
                rows.append(primary_v + api_v + eng_v)

            audio_mat = np.array(rows, dtype="float32")
            scaler    = bundle.get("audio_scaler")
            if scaler:
                try:
                    audio_mat = scaler.transform(audio_mat)
                except Exception:
                    pass

            X_pool = hstack([
                csr_matrix((len(pool), n_tfidf)),
                csr_matrix(np.zeros((len(pool), 4), dtype="float32")),
                csr_matrix(audio_mat),
            ])

            proba_all  = model.predict_proba(X_pool)
            pred_idx   = np.argmax(proba_all, axis=1)
            classes    = le.classes_ if le is not None else list(EMOTION_STYLE.keys())
            predicted_emotions = (le.inverse_transform(pred_idx) if le is not None
                                  else np.array(classes)[pred_idx])

            pool["_pred_em"] = predicted_emotions
            for i, em in enumerate(classes):
                pool[f"_conf_{em}"] = proba_all[:, i]

        except Exception:
            predicted_emotions = None

    top2 = sorted(emotion_weights, key=emotion_weights.get, reverse=True)[:2]

    if predicted_emotions is not None and "_pred_em" in pool.columns:
        candidates = pool[pool["_pred_em"].isin(top2)].copy()
        if len(candidates) < n_songs:
            candidates = pool[pool["_pred_em"] == dominant_em].copy()
        if len(candidates) < n_songs:
            candidates = pool.copy()
    else:
        # Fallback: filter by BERT label when model not available
        candidates = pool[pool["emotion_4"].isin(top2)].copy()

    def audio_dist(row):
        d = 0.0
        for col in audio_cols:
            if col in row.index:
                v   = float(row[col]) if pd.notna(row.get(col)) else 0.0
                tgt = target_audio.get(col, 0.5)
                if col == "Tempo":
                    v /= 200; tgt /= 200
                elif col == "Loudness":
                    v = (v + 60) / 60; tgt = (tgt + 60) / 60
                d += (v - tgt) ** 2
        return d ** 0.5

    candidates["_audio_dist"] = candidates.apply(audio_dist, axis=1)
    return candidates.nsmallest(n_songs, "_audio_dist").copy(), emotion_weights, dominant_em


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP — load everything
# ══════════════════════════════════════════════════════════════════════════════

bundle   = load_models()
metadata = load_metadata()

# Dataset — try disk first, then fall back to sidebar uploader
df = load_dataset()

if df is None:
    st.warning(
        "⚠️ Dataset not found at `data/spotify_dataset_withemotion.csv`.\n\n"
        "Upload it below (the app will process it in memory for this session)."
    )
    uploaded = st.sidebar.file_uploader(
        "Upload spotify_dataset_withemotion.csv",
        type="csv",
        key="dataset_upload",
    )
    if uploaded is not None:
        try:
            _df = pd.read_csv(uploaded)
            if "emotion_4" not in _df.columns and "emotion" in _df.columns:
                _df["emotion"] = _df["emotion"].replace({"Love": "love", "angry": "anger"})
                _df["emotion_4"] = _df["emotion"].map(EMOTION_4_MAP)
            valid = list(EMOTION_STYLE.keys())
            _df   = _df[_df["emotion_4"].isin(valid)].copy()
            if "Artist(s)" in _df.columns and "song" in _df.columns:
                _df = _df.drop_duplicates(subset=["Artist(s)", "song"], keep="first")
            for col in PRIMARY_AUDIO_COLS:
                if col in _df.columns:
                    _df[col] = pd.to_numeric(_df[col], errors="coerce")
            df = _df.reset_index(drop=True)
            st.success(f"✅ Dataset loaded from upload — {len(df):,} tracks.")
        except Exception as exc:
            st.error(f"❌ Upload failed: {exc}")

if df is None:
    st.error("Cannot run without a dataset. Please add `data/spotify_dataset_withemotion.csv` or upload it.")
    st.stop()

# Available classifiers
AVAILABLE_MODELS = [n for n in MODEL_FILES if bundle and n in bundle]
perf             = metadata.get("model_performance", {})

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎵 MoodSense")
    st.markdown("**Music Emotion Classifier**")
    st.markdown("---")

    st.markdown("#### 📊 Dataset")
    st.metric("Tracks", f"{len(df):,}")
    ec = df["emotion_4"].value_counts()
    for em, cnt in ec.items():
        pct = cnt / len(df) * 100
        st.markdown(f"{EMOTION_STYLE[em]['emoji']} **{em.title()}** — {pct:.1f}%")

    st.markdown("---")
    st.markdown("#### 🤖 Active Model")

    if AVAILABLE_MODELS:
        def _lbl(n):
            p   = perf.get(n.lower(), {})
            acc = p.get("accuracy")
            return f"{n} ({acc*100:.1f}%)" if acc else n

        model_choice_label = st.selectbox("Model", [_lbl(m) for m in AVAILABLE_MODELS])
        model_key          = model_choice_label.split(" ")[0]
    else:
        st.error("No classifier models found in model/")
        model_key = None

    st.markdown("---")
    st.markdown("#### 📚 Project")
    st.markdown("**Course:** MS DSP 422\n\n**Team:** Group 3")
    if metadata:
        st.caption(f"Trained: {metadata.get('training_date', '—')}")
        st.caption(f"Features: {metadata.get('num_features', '—'):,}" if isinstance(metadata.get("num_features"), int) else "")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1 class="hero-title">🎵 MoodSense</h1>
  <p class="hero-sub">AI-Powered Music Emotion Classification · MS DSP 422 Group 3</p>
</div>
""", unsafe_allow_html=True)

best_acc = max((v.get("accuracy", 0) for v in perf.values()), default=0)
best_f1  = max((v.get("f1_weighted", 0) for v in perf.values()), default=0)
n_feat   = metadata.get("num_features", "—")

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df):,}",                                  "Dataset Tracks"),
    (c2, f"{best_acc*100:.1f}%" if best_acc else "—",     "Best Accuracy"),
    (c3, f"{best_f1:.4f}"       if best_f1  else "—",     "Best Weighted F1"),
    (c4, f"{n_feat:,}" if isinstance(n_feat, int) else str(n_feat), "Total Features"),
]:
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
    "🎵 Explore Dataset",
    "📊 Analytics",
    "🔬 Model Performance",
    "🎧 Your Mood, Your Playlist",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLASSIFY SONG
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🎯 Classify a Song's Emotion")

    if not AVAILABLE_MODELS or model_key is None:
        st.error("No trained models available. Add pkl files to the model/ directory.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Song Info")
            song_name   = st.text_input("Song Title *",  placeholder="e.g. Bohemian Rhapsody")
            artist_name = st.text_input("Artist *",       placeholder="e.g. Queen")
            lyrics      = st.text_area(
                "Lyrics (optional — improves accuracy)",
                placeholder="Paste lyrics here…", height=170,
            )

        with col_b:
            st.subheader("Audio Features")
            energy   = st.slider("⚡ Energy",              0.0, 1.0, 0.65, 0.01)
            dance    = st.slider("💃 Danceability",        0.0, 1.0, 0.60, 0.01)
            pos      = st.slider("😊 Positiveness",        0.0, 1.0, 0.50, 0.01)
            acoustic = st.slider("🎸 Acousticness",        0.0, 1.0, 0.30, 0.01)
            tempo    = st.slider("🥁 Tempo (BPM)",         60,  200, 120,  1)
            speech   = st.slider("🗣 Speechiness",         0.0, 1.0, 0.05, 0.01)
            instr    = st.slider("🎹 Instrumentalness",    0.0, 1.0, 0.10, 0.01)
            loudness = st.slider("🔊 Loudness (dB)",      -60.0, 0.0, -7.0, 0.5)
            pop      = st.slider("⭐ Popularity",          0,   100, 50,   1)

        st.markdown("---")
        go = st.button("🎵 Classify Emotion", type="primary", use_container_width=True)

        if go:
            if not song_name or not artist_name:
                st.warning("Please enter a song title and artist name.")
            else:
                with st.spinner("Analysing emotion…"):
                    song_vals = {
                        "Energy": energy, "Danceability": dance, "Positiveness": pos,
                        "Acousticness": acoustic, "Tempo": tempo, "Speechiness": speech,
                        "Instrumentalness": instr, "Loudness": loudness,
                        "Popularity": pop, "Liveness": 0.1,
                    }
                    X   = build_feature_vector(song_vals, lyrics, bundle)
                    res = predict(X, bundle, model_key)

                if res is None:
                    st.error("Prediction failed — check model logs.")
                else:
                    em    = res["prediction"]
                    conf  = res["confidence"]
                    probs = res["probabilities"]
                    style = EMOTION_STYLE.get(em, {"color": "#666", "emoji": "🎵", "desc": ""})

                    st.success("✅ Classification complete!")
                    r1, r2 = st.columns(2)

                    with r1:
                        st.markdown(
                            f'<div class="emo-card" style="background:{style["color"]};color:white;">'
                            f'<div class="emo-emoji">{style["emoji"]}</div>'
                            f'<div class="emo-name">{em.upper()}</div>'
                            f'<div class="emo-desc">{style["desc"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        level = "High" if conf > .65 else "Medium" if conf > .45 else "Low"
                        st.metric("Model Confidence", f"{conf*100:.1f}%", delta=level)
                        st.caption(f"Model: {model_key}")

                    with r2:
                        st.subheader("Probability Distribution")
                        prob_df = pd.DataFrame({
                            "Emotion":     [e.title() for e in probs],
                            "Probability": list(probs.values()),
                        }).sort_values("Probability", ascending=False)

                        colors = {e.title(): EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE}
                        fig = px.bar(
                            prob_df, x="Emotion", y="Probability",
                            color="Emotion", color_discrete_map=colors, text="Probability",
                        )
                        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                        fig.update_layout(
                            showlegend=False, height=320, yaxis_range=[0, 1],
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.subheader(f"🎵 Similar {em.title()} Songs in Dataset")
                    sim = similar_songs(song_vals, df, em, n=10)

                    if not sim.empty:
                        sim_display = sim.rename(columns={
                            "song": "🎵 Song", "Artist(s)": "🎤 Artist",
                            "Energy": "⚡ Energy", "Positiveness": "😊 Positivity",
                            "_dist": "📏 Distance",
                        })
                        sim_display["📏 Distance"] = sim_display["📏 Distance"].round(4)
                        st.dataframe(
                            sim_display[["🎵 Song", "🎤 Artist", "⚡ Energy",
                                         "😊 Positivity", "📏 Distance"]],
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("No similar songs found.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLORE DATASET
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🎵 Explore the Dataset")

    sel_em = st.selectbox(
        "Filter by emotion:",
        ["All"] + list(EMOTION_STYLE.keys()),
        format_func=lambda x: (
            f"{EMOTION_STYLE[x]['emoji']} {x.title()}" if x != "All" else "All Emotions"
        ),
    )

    fdf = df if sel_em == "All" else df[df["emotion_4"] == sel_em]
    st.info(f"Showing {len(fdf):,} songs")

    disp_cols  = [c for c in ["song", "Artist(s)", "emotion_4", "Energy",
                               "Positiveness", "Danceability"] if c in fdf.columns]
    rename_map = {
        "song": "🎵 Song", "Artist(s)": "🎤 Artist", "emotion_4": "🎭 Emotion",
        "Energy": "⚡ Energy", "Positiveness": "😊 Positivity", "Danceability": "💃 Dance",
    }
    st.dataframe(
        fdf[disp_cols].head(200).rename(columns=rename_map),
        use_container_width=True, height=500,
    )

    csv = fdf[disp_cols].to_csv(index=False).encode()
    st.download_button(
        f"📥 Download CSV ({len(fdf):,} songs)", csv,
        f"moodsense_{sel_em}.csv", "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Emotion Analytics")

    ec = df["emotion_4"].value_counts()
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = px.pie(
            values=ec.values,
            names=[e.title() for e in ec.index],
            color=ec.index,
            color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in ec.index},
            hole=.4, title="Emotion Distribution",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Breakdown")
        for em, cnt in ec.items():
            pct = cnt / len(df) * 100
            st.markdown(f"{EMOTION_STYLE[em]['emoji']} **{em.title()}**: {cnt:,} ({pct:.1f}%)")

    st.markdown("---")
    st.subheader("📈 Audio Features by Emotion")
    avail_feat = [c for c in ["Energy", "Positiveness", "Danceability", "Acousticness", "Tempo"]
                  if c in df.columns]
    feat = st.selectbox("Feature:", avail_feat)
    fig2 = px.violin(
        df, x="emotion_4", y=feat, color="emotion_4",
        color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
        box=True, points=False, labels={"emotion_4": "Emotion", feat: feat},
    )
    fig2.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Feature Scatter by Emotion")
    avail2 = [c for c in ["Energy", "Positiveness", "Danceability", "Acousticness"] if c in df.columns]
    if len(avail2) >= 2:
        fx = st.selectbox("X-axis:", avail2, index=0)
        fy = st.selectbox("Y-axis:", avail2, index=min(1, len(avail2)-1))
        fig3 = px.scatter(
            df.sample(min(5_000, len(df)), random_state=42),
            x=fx, y=fy, color="emotion_4",
            color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
            opacity=.5, height=420,
        )
        st.plotly_chart(fig3, use_container_width=True)


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
        k4.metric("Training Samples",
                  f"{metadata.get('training_samples','—'):,}"
                  if isinstance(metadata.get("training_samples"), int) else "—")

        st.markdown("---")
        st.subheader("📊 Model Comparison (from model_metadata.json)")

        perf_df = pd.DataFrame([
            {"Model": m, "Accuracy": v.get("accuracy", 0), "Weighted F1": v.get("f1_weighted", 0)}
            for m, v in perf.items()
        ]).sort_values("Accuracy", ascending=False)

        fig = px.bar(
            perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Performance Comparison",
            color_discrete_sequence=["#1d4ed8", "#3b82f6"],
        )
        fig.update_layout(height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Breakdown")
        fb = metadata.get("feature_breakdown", {})
        if fb:
            fb_df  = pd.DataFrame({"Feature Group": list(fb.keys()), "Count": list(fb.values())})
            fb_fig = px.bar(
                fb_df, x="Feature Group", y="Count", color="Feature Group",
                color_discrete_sequence=["#0f172a", "#1d4ed8", "#3b82f6"],
            )
            fb_fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fb_fig, use_container_width=True)
        else:
            st.info("Feature breakdown not found in metadata.")


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
            Describe a vibe, a moment, a feeling. The ML pipeline reads your words, maps them to
            emotion space via VADER + TF-IDF cosine similarity, then runs the trained classifier
            on the dataset to curate a playlist. No rules — all signal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    centroids = get_audio_centroids(df)

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
                st.session_state["playlist_prompt"] = p["prompt"]
                st.session_state["playlist_n"]      = p["n"]
                st.session_state["playlist_title"]  = f"{p['icon']} {p['title']}"
                st.session_state["run_playlist"]    = True

    with col_custom:
        st.markdown("#### ✍️ Describe Your Own Vibe")
        st.caption("Type anything — a feeling, a scene, a moment.")
        prompt_input = st.text_area(
            "Your mood prompt",
            placeholder=(
                'e.g. "Sunday morning coffee, soft rain on the window, '
                'feeling nostalgic but okay…"'
            ),
            height=110,
            label_visibility="collapsed",
            key="custom_prompt_input",
        )
        ca, cb, cc = st.columns([2, 1, 1])
        with ca:
            n_songs_custom = st.slider("Playlist length", 5, 40, 20, 5, key="n_slider")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎵 Generate", type="primary", use_container_width=True, key="gen_custom"):
                if prompt_input.strip():
                    st.session_state["playlist_prompt"] = prompt_input.strip()
                    st.session_state["playlist_n"]      = n_songs_custom
                    st.session_state["playlist_title"]  = "🎧 Custom Playlist"
                    st.session_state["run_playlist"]    = True
                else:
                    st.warning("Please describe a mood or vibe first.")
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear", use_container_width=True, key="clear_pl"):
                for k in ["playlist_prompt", "playlist_n", "playlist_title",
                          "run_playlist", "pl_result"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Run generator
    if st.session_state.get("run_playlist"):
        st.session_state["run_playlist"] = False
        active_prompt = st.session_state.get("playlist_prompt", "")
        active_n      = st.session_state.get("playlist_n", 20)
        active_title  = st.session_state.get("playlist_title", "Your Playlist")

        st.markdown("---")
        with st.spinner("🔮 Classifying songs through the ML model…"):
            pl_df, emo_weights, dominant = generate_playlist(
                prompt_text=active_prompt,
                df=df,
                bundle=bundle,
                model_key=model_key,
                n_songs=active_n,
                centroids=centroids,
            )
            st.session_state["pl_result"] = {
                "df": pl_df, "weights": emo_weights,
                "dominant": dominant, "title": active_title,
                "prompt": active_prompt,
            }

    # Render result
    res = st.session_state.get("pl_result")
    if res and not res["df"].empty:
        pl_df     = res["df"]
        weights   = res["weights"]
        dominant  = res["dominant"]
        pl_title  = res["title"]
        pl_prompt = res["prompt"]

        st.markdown("---")
        dom_style = EMOTION_STYLE.get(dominant, {"color": "#1d4ed8", "emoji": "🎵"})
        st.markdown(
            f'<div class="playlist-card" style="border-top:4px solid {dom_style["color"]};">'
            f'<div class="playlist-header">{pl_title}</div>'
            f'<div class="playlist-meta">Prompt: <em>"{pl_prompt}"</em> &nbsp;·&nbsp; '
            f'{len(pl_df)} tracks &nbsp;·&nbsp; Dominant: '
            f'<strong>{dominant.title()}</strong> {dom_style["emoji"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        rr1, rr2 = st.columns([1, 2])
        with rr1:
            st.markdown("**NLP Emotion Signal**")
            st.caption("VADER + TF-IDF cosine similarity mapped to emotion space")
            w_df = pd.DataFrame({
                "Emotion": [e.title() for e in weights],
                "Weight":  list(weights.values()),
            }).sort_values("Weight")
            fig_w = px.bar(
                w_df, x="Weight", y="Emotion", orientation="h",
                color="Emotion",
                color_discrete_map={e.title(): EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                text="Weight",
            )
            fig_w.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_w.update_layout(
                showlegend=False, height=220,
                xaxis_range=[0, max(weights.values()) * 1.35],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=40, t=10, b=10),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with rr2:
            st.markdown("**Playlist Audio Profile**")
            st.caption("Songs coloured by ML-predicted emotion")
            color_col = "_pred_em" if "_pred_em" in pl_df.columns else "emotion_4"
            if "Energy" in pl_df.columns and "Positiveness" in pl_df.columns:
                sc_fig = px.scatter(
                    pl_df, x="Energy", y="Positiveness",
                    color=color_col,
                    color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                    hover_data=["song", "Artist(s)"] if "song" in pl_df.columns else [],
                    opacity=.85, height=220,
                )
                sc_fig.update_layout(
                    showlegend=True, margin=dict(l=0, r=0, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(sc_fig, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### 🎵 Tracklist — {len(pl_df)} songs")

        conf_col = f"_conf_{dominant}" if f"_conf_{dominant}" in pl_df.columns else None
        for idx, (_, row) in enumerate(pl_df.iterrows(), 1):
            song_em  = row.get("_pred_em", row.get("emotion_4", dominant))
            em_style = EMOTION_STYLE.get(str(song_em), {"color": "#64748b", "emoji": "🎵"})
            conf_val = float(row[conf_col]) if conf_col and pd.notna(row.get(conf_col)) else None
            conf_pct = int(conf_val * 100) if conf_val else 0

            st.markdown(
                f'<div class="track-row">'
                f'<span class="track-num">{idx}</span>'
                f'<div class="track-info">'
                f'<div class="track-name">{row.get("song","—")}</div>'
                f'<div class="track-artist">{row.get("Artist(s)","—")}</div>'
                f'</div>'
                f'<span class="track-badge" '
                f'style="background:{em_style["color"]}22;color:{em_style["color"]};">'
                f'{em_style["emoji"]} {str(song_em).title()}</span>'
                f'<div class="conf-bar-wrap" title="Model confidence: {conf_pct}%">'
                f'<div class="conf-bar-bg">'
                f'<div class="conf-bar-fg" '
                f'style="width:{conf_pct}%;background:{em_style["color"]};"></div>'
                f'</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        export_cols = [c for c in ["song", "Artist(s)", "_pred_em", "Energy",
                                    "Positiveness", "Danceability", "_audio_dist"]
                       if c in pl_df.columns]
        st.download_button(
            f"📥 Export Playlist CSV ({len(pl_df)} songs)",
            pl_df[export_cols].rename(columns={
                "_pred_em": "ML Predicted Emotion",
                "_audio_dist": "Audio Distance to Target",
            }).to_csv(index=False).encode(),
            "moodsense_playlist.csv", "text/csv",
            use_container_width=True,
        )

    elif st.session_state.get("pl_result") and res["df"].empty:
        st.warning("No songs matched. Try a different prompt or check that models are loaded.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748b;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎶</div>
            <p style="font-size:1.1rem;font-weight:600;">
                Pick a preset vibe on the left or type your own mood above.
            </p>
            <p style="font-size:.9rem;">
                The ML model classifies songs from the dataset and curates
                a playlist ranked by audio similarity to your emotional signal.
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
