"""
MoodSense: Music Emotion Classification Platform
─────────────────────────────────────────────────
MS DSP 422 - Practical Machine Learning | Group 3

Dataset source: Google BigQuery
    project  : probable-scout-477715-k7
    dataset  : Moodsense
    table    : spotify_dataset_withemotion

Repo layout:
    moodsense-app/
    ├── moodsense_app.py        ← this file
    ├── requirements.txt
    └── model/
        ├── audio_scaler.pkl
        ├── vader_scaler.pkl    (optional)
        ├── tfidf.pkl
        ├── label_encoder.pkl
        ├── lightgbm_model.pkl
        ├── xgboost_model.pkl
        ├── ensemble_model.pkl
        ├── linearsvc_model.pkl
        ├── logistic_model.pkl
        └── model_metadata.json

Auth — Streamlit Cloud (.streamlit/secrets.toml):
    [gcp_service_account]
    type                        = "service_account"
    project_id                  = "probable-scout-477715-k7"
    private_key_id              = "..."
    private_key                 = "-----BEGIN RSA PRIVATE KEY-----\\n..."
    client_email                = "...@....iam.gserviceaccount.com"
    client_id                   = "..."
    auth_uri                    = "https://accounts.google.com/o/oauth2/auth"
    token_uri                   = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url        = "..."

    Service account roles: BigQuery Data Viewer + BigQuery Job User

Auth — local dev (either):
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    OR: gcloud auth application-default login

Run:
    pip install -r requirements.txt
    streamlit run moodsense_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
# WeightedEnsemble — MUST be defined before any pickle.load()
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
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder.inverse_transform(idx)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS — relative to this file's directory (repo root)
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = Path(__file__).parent.resolve()

MODEL_DIR          = _ROOT / "model"
AUDIO_SCALER_PATH  = MODEL_DIR / "audio_scaler.pkl"
VADER_SCALER_PATH  = MODEL_DIR / "vader_scaler.pkl"
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
# BIGQUERY CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BQ_PROJECT  = "probable-scout-477715-k7"
BQ_TABLE_ID = "probable-scout-477715-k7.Moodsense.spotify_dataset_withemotion"

# Emotion mapping + dedup done server-side in BigQuery to minimise data transfer.
# ROW_NUMBER deduplicates on (song, Artist(s)), preferring the row with most lyrics.
BQ_QUERY = """
SELECT
    song,
    `Artist(s)`                                                         AS artist,
    text                                                                AS lyrics,
    SAFE_CAST(Energy           AS FLOAT64)                              AS Energy,
    SAFE_CAST(Danceability     AS FLOAT64)                              AS Danceability,
    SAFE_CAST(Positiveness     AS FLOAT64)                              AS Positiveness,
    SAFE_CAST(Speechiness      AS FLOAT64)                              AS Speechiness,
    SAFE_CAST(Liveness         AS FLOAT64)                              AS Liveness,
    SAFE_CAST(Acousticness     AS FLOAT64)                              AS Acousticness,
    SAFE_CAST(Instrumentalness AS FLOAT64)                              AS Instrumentalness,
    SAFE_CAST(Tempo            AS FLOAT64)                              AS Tempo,
    SAFE_CAST(
        REGEXP_REPLACE(CAST(Loudness AS STRING), r'[^0-9.\\-]', '')
    AS FLOAT64)                                                         AS Loudness,
    SAFE_CAST(Popularity       AS FLOAT64)                              AS Popularity,
    CASE
        WHEN LOWER(emotion) IN ('joy',     'surprise') THEN 'happy'
        WHEN LOWER(emotion) IN ('sadness', 'fear')     THEN 'sad'
        WHEN LOWER(emotion) IN ('anger',   'angry')    THEN 'anger'
        WHEN emotion         IN ('love',   'Love')     THEN 'love'
        ELSE NULL
    END                                                                 AS emotion_4
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY song, `Artist(s)`
            ORDER BY LENGTH(IFNULL(text, '')) DESC
        ) AS _rn
    FROM `{table}`
    WHERE emotion IS NOT NULL
      AND text    IS NOT NULL
      AND LENGTH(IFNULL(text, '')) > 50
)
WHERE _rn = 1
  AND emotion IN ('joy','surprise','sadness','fear','anger','angry','love','Love')
LIMIT 550000
""".format(table=BQ_TABLE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_STYLE = {
    "happy": {"color": "#10b981", "emoji": "😊", "desc": "Joyful and uplifting"},
    "sad":   {"color": "#3b82f6", "emoji": "😢", "desc": "Melancholic and reflective"},
    "anger": {"color": "#ef4444", "emoji": "🔥", "desc": "Intense and powerful"},
    "love":  {"color": "#ec4899", "emoji": "💕", "desc": "Romantic and tender"},
}

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

AUDIO_RANGES = {
    "Energy":           (0.0,  1.0,  0.65),
    "Danceability":     (0.0,  1.0,  0.60),
    "Positiveness":     (0.0,  1.0,  0.50),
    "Speechiness":      (0.0,  1.0,  0.05),
    "Liveness":         (0.0,  1.0,  0.12),
    "Acousticness":     (0.0,  1.0,  0.30),
    "Instrumentalness": (0.0,  1.0,  0.10),
    "Tempo":            (60.0, 200.0, 120.0),
    "Loudness":         (-60.0, 0.0,  -7.0),
    "Popularity":       (0.0,  100.0, 50.0),
}

# Fallback centroids used only when BQ dataset is unavailable
EMOTION_CENTROIDS_FALLBACK = {
    "happy": {"Energy":0.75,"Danceability":0.72,"Positiveness":0.78,"Speechiness":0.07,
              "Liveness":0.17,"Acousticness":0.22,"Instrumentalness":0.04,
              "Tempo":128,"Loudness":-5,"Popularity":65},
    "sad":   {"Energy":0.38,"Danceability":0.42,"Positiveness":0.25,"Speechiness":0.04,
              "Liveness":0.12,"Acousticness":0.60,"Instrumentalness":0.08,
              "Tempo":98, "Loudness":-10,"Popularity":48},
    "anger": {"Energy":0.82,"Danceability":0.55,"Positiveness":0.35,"Speechiness":0.12,
              "Liveness":0.18,"Acousticness":0.15,"Instrumentalness":0.05,
              "Tempo":145,"Loudness":-4,"Popularity":55},
    "love":  {"Energy":0.52,"Danceability":0.60,"Positiveness":0.72,"Speechiness":0.05,
              "Liveness":0.10,"Acousticness":0.40,"Instrumentalness":0.03,
              "Tempo":108,"Loudness":-7,"Popularity":60},
}

PRESET_PLAYLISTS = [
    {"id":"work_focus",       "icon":"💼","title":"Deep Work Focus",
     "subtitle":"Instrumental · low distraction · steady tempo",
     "prompt":"instrumental focus concentration work study calm steady beats no lyrics","n":20},
    {"id":"rainy_afternoon",  "icon":"🌧️","title":"Rainy Afternoon",
     "subtitle":"Melancholic · acoustic · introspective",
     "prompt":"rain melancholic quiet acoustic reflective slow afternoon sad gentle","n":18},
    {"id":"heartbreak",       "icon":"💔","title":"Heartbreak Hotel",
     "subtitle":"Emotional · raw · post-breakup vibes",
     "prompt":"heartbreak crying pain love lost breakup emotional tears missing someone","n":18},
    {"id":"late_night_drive", "icon":"🚗","title":"Late Night Drive",
     "subtitle":"Dark · atmospheric · driving energy",
     "prompt":"night dark driving highway electric atmospheric moody intensity beat","n":20},
    {"id":"summer_party",     "icon":"🌞","title":"Summer Party",
     "subtitle":"High energy · danceable · joyful",
     "prompt":"party dance happy upbeat energy summer fun celebration joyful loud","n":20},
    {"id":"romance",          "icon":"🌹","title":"Date Night",
     "subtitle":"Romantic · tender · warm",
     "prompt":"love romance tender sweet gentle warmth affection intimate caring","n":18},
    {"id":"anger_release",    "icon":"⚡","title":"Rage Release",
     "subtitle":"Intense · powerful · cathartic",
     "prompt":"anger intense powerful aggressive rage frustration electric loud cathartic","n":18},
    {"id":"morning_run",      "icon":"🏃","title":"Morning Run",
     "subtitle":"Energetic · motivating · fast tempo",
     "prompt":"run exercise morning energetic motivating fast beats workout pump adrenaline","n":20},
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
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
.kpi-val { font-size:2.2rem; font-weight:800; color:var(--blue);
           margin:0; font-family:'Syne',sans-serif; }
.kpi-lbl { font-size:.75rem; color:var(--muted); text-transform:uppercase;
           letter-spacing:.08em; font-weight:600; margin-top:.4rem; }

.emo-card  { padding:2.5rem; border-radius:18px; text-align:center;
             box-shadow:0 12px 40px rgba(0,0,0,.18); margin:1.5rem 0; }
.emo-emoji { font-size:5rem; }
.emo-name  { font-size:2.2rem; font-weight:800; margin:.8rem 0 .4rem;
             font-family:'Syne',sans-serif; }
.emo-desc  { font-size:1rem; opacity:.85; }

.playlist-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:16px; padding:1.75rem;
    box-shadow:0 4px 16px rgba(0,0,0,.06); margin-bottom:1rem;
}
.playlist-header { font-family:'Syne',sans-serif; font-size:1.6rem;
                   font-weight:800; margin:0 0 .3rem; }
.playlist-meta   { color:var(--muted); font-size:.85rem; margin-bottom:1rem; }

.track-row {
    display:flex; align-items:center; gap:1rem; padding:.65rem .5rem;
    border-radius:8px; border-bottom:1px solid var(--border); transition:background .15s;
}
.track-row:hover  { background:var(--bg); }
.track-num    { width:1.8rem; text-align:right; color:var(--muted);
                font-size:.85rem; font-weight:600; flex-shrink:0; }
.track-info   { flex:1; min-width:0; }
.track-name   { font-weight:600; white-space:nowrap;
                overflow:hidden; text-overflow:ellipsis; }
.track-artist { font-size:.8rem; color:var(--muted); }
.track-badge  { font-size:.72rem; font-weight:700; padding:.2rem .65rem;
                border-radius:999px; flex-shrink:0; }
.conf-wrap { width:80px; flex-shrink:0; }
.conf-bg   { background:#e2e8f0; border-radius:4px; height:6px; }
.conf-fg   { border-radius:4px; height:6px; }

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
# BIGQUERY AUTH
# ══════════════════════════════════════════════════════════════════════════════
def _get_bq_credentials():
    """
    Returns GCP credentials. Priority:
      1. st.secrets["gcp_service_account"]  — Streamlit Cloud / secrets.toml
      2. Application Default Credentials    — env var or gcloud auth
    """
    try:
        from google.oauth2 import service_account
        if "gcp_service_account" in st.secrets:
            return service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
            )
    except Exception:
        pass

    try:
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/bigquery.readonly"]
        )
        return creds
    except Exception as exc:
        raise RuntimeError(
            "No GCP credentials found.\n\n"
            "• Streamlit Cloud: add [gcp_service_account] to .streamlit/secrets.toml\n"
            "• Local: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json\n"
            "• Local: run  gcloud auth application-default login\n\n"
            f"Original error: {exc}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    """Load all pkl artifacts from model/. Returns bundle dict or None."""
    bundle = {}

    def _load(p: Path):
        with open(p, "rb") as f:
            return pickle.load(f)

    missing = [str(p) for p in [AUDIO_SCALER_PATH, TFIDF_PATH, LABEL_ENCODER_PATH]
               if not p.exists()]
    if missing:
        st.error(
            "Required model files missing:\n" +
            "\n".join(f"  • {m}" for m in missing)
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
        return bundle
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(
    show_spinner="📡 Fetching dataset from BigQuery…",
    ttl=3600,   # re-query at most once per hour
)
def load_dataset_from_bq():
    """
    Pull emotion-mapped, deduplicated song data from BigQuery.
    Deduplication and emotion_4 mapping are handled inside the SQL query.

    Returns DataFrame with columns:
        song, artist, lyrics, emotion_4,
        Energy, Danceability, Positiveness, Speechiness, Liveness,
        Acousticness, Instrumentalness, Tempo, Loudness, Popularity

    Returns None on failure (caller will st.stop()).
    """
    try:
        from google.cloud import bigquery

        creds  = _get_bq_credentials()
        client = bigquery.Client(project=BQ_PROJECT, credentials=creds)
        df     = client.query(BQ_QUERY).to_dataframe()

        # Keep only rows where the CASE WHEN matched a valid emotion_4
        df = df[df["emotion_4"].isin(list(EMOTION_STYLE.keys()))].copy()

        # Coerce audio columns to numeric (edge-case NaN safety)
        for col in PRIMARY_AUDIO_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.reset_index(drop=True)

    except ImportError:
        st.error(
            "`google-cloud-bigquery` is not installed.\n\n"
            "Run:  pip install google-cloud-bigquery pandas-gbq"
        )
        return None
    except RuntimeError as exc:
        st.error(f"GCP credentials error:\n\n{exc}")
        return None
    except Exception as exc:
        st.error(f"BigQuery query failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE — mirrors training exactly
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
        10 primary + 12 API zeros (unavailable at inference) + 7 engineered
    """
    def sv(k, d=0.0):
        return float(vals.get(k, d))

    energy   = sv("Energy",           0.5)
    dance    = sv("Danceability",      0.5)
    pos      = sv("Positiveness",      0.5)
    speech   = sv("Speechiness",      0.05)
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
        0.0,                     # valence_dance  (API cols unavailable at inference)
        0.0,                     # arousal
    ]
    return primary + api_zeros + engineered


def build_feature_matrix(audio_rows: list, lyrics_list: list, bundle: dict):
    """
    Build sparse (n, 25033) matrix: TF-IDF(25000) + VADER(4) + Audio(29)
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
    Returns (string_labels: np.ndarray, proba_matrix: np.ndarray).
    Handles both predict_proba and decision-only models (LinearSVC).
    """
    model = bundle.get(model_name)
    le    = bundle.get("label_encoder")
    if model is None:
        raise ValueError(f"Model '{model_name}' not in bundle.")

    if hasattr(model, "predict_proba"):
        proba    = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        classes  = le.classes_ if le is not None else list(EMOTION_STYLE.keys())
        labels   = (np.array([str(le.inverse_transform([i])[0]) for i in pred_idx])
                    if le is not None
                    else np.array([str(classes[i]) for i in pred_idx]))
    else:
        raw    = model.predict(X)
        labels = np.array([str(r) for r in raw])
        cls    = list(EMOTION_STYLE.keys())
        proba  = np.zeros((len(labels), len(cls)), dtype="float32")
        for i, lbl in enumerate(labels):
            if lbl in cls:
                proba[i, cls.index(lbl)] = 1.0

    return labels, proba


# ══════════════════════════════════════════════════════════════════════════════
# PLAYLIST ENGINE — NLP prompt → ML classifier → real BigQuery songs
# ══════════════════════════════════════════════════════════════════════════════
def parse_prompt_to_weights(prompt: str, bundle: dict) -> dict:
    """
    NLP: free-text → soft emotion weights.
    VADER polarity + TF-IDF cosine similarity against emotion reference docs.
    Purely continuous — no threshold rules.
    """
    vader_obj             = SentimentIntensityAnalyzer()
    vs                    = vader_obj.polarity_scores(prompt)
    compound, neg, pos_   = vs["compound"], vs["neg"], vs["pos"]

    vader_sig = {
        "happy": pos_ * (1 + compound) / 2,
        "love":  pos_ * max(0.0, 1 - neg),
        "sad":   neg  * (2 - compound) / 2,
        "anger": neg  * (1 + abs(compound)),
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
                d_vec         = tfidf.transform([clean_text(doc)])
                tfidf_sig[em] = float(cosine_similarity(p_vec, d_vec)[0][0])
        except Exception:
            pass

    raw   = {em: 0.5 * vader_sig.get(em, 0.0) + 0.5 * tfidf_sig[em] for em in EMOTION_STYLE}
    total = sum(raw.values()) or 1.0
    return {em: v / total for em, v in raw.items()}


@st.cache_data(show_spinner=False)
def compute_bq_centroids(_df: pd.DataFrame) -> dict:
    """Mean audio feature vector per emotion, computed from actual BigQuery data."""
    audio_cols = [c for c in PRIMARY_AUDIO_COLS if c in _df.columns]
    return {
        em: _df[_df["emotion_4"] == em][audio_cols].mean().to_dict()
        for em in EMOTION_STYLE
    }


def generate_playlist(
    prompt: str,
    df: pd.DataFrame,
    bundle: dict,
    model_key: str,
    n_songs: int = 20,
) -> tuple:
    """
    Playlist pipeline — strictly ML/NLP on real BigQuery songs.

    1. NLP → soft emotion weights (VADER + TF-IDF cosine)
    2. Weighted audio centroid from BQ data → target audio vector
    3. Sample 5 000 real BQ songs as candidate pool
    4. ML model predicts emotion for each candidate (audio features only)
    5. Filter to top-2 predicted emotions → rank by audio distance → top-n
    """
    weights  = parse_prompt_to_weights(prompt, bundle)
    dominant = max(weights, key=weights.get)

    # Centroids from actual BQ data
    centroids  = (compute_bq_centroids(df)
                  if df is not None and not df.empty
                  else EMOTION_CENTROIDS_FALLBACK)
    audio_cols = [c for c in PRIMARY_AUDIO_COLS if c in df.columns]

    target_audio = {
        col: sum(weights[em] * centroids.get(em, {}).get(col, AUDIO_RANGES[col][2])
                 for em in weights)
        for col in audio_cols
    }

    # Sample pool of real songs from BigQuery dataset
    pool      = df.sample(n=min(5_000, len(df)), random_state=42).copy()
    audio_rows = [
        {c: (float(row[c]) if pd.notna(row.get(c)) else AUDIO_RANGES[c][2])
         for c in PRIMARY_AUDIO_COLS if c in pool.columns}
        for _, row in pool.iterrows()
    ]

    try:
        tfidf_obj = bundle.get("tfidf")
        n_tfidf   = tfidf_obj.get_feature_names_out().shape[0] if tfidf_obj else 25_000
        audio_mat = np.array([audio_vals_to_vector(r) for r in audio_rows], dtype="float32")
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

        pred_labels, proba_all = run_inference(X_pool, bundle, model_key)
        le      = bundle.get("label_encoder")
        classes = list(le.classes_) if le is not None else list(EMOTION_STYLE.keys())

        pool["_pred_em"]    = pred_labels
        pool["_confidence"] = proba_all.max(axis=1)
        for i, em in enumerate(classes):
            pool[f"_conf_{em}"] = proba_all[:, i]

    except Exception as exc:
        st.error(f"Playlist inference error: {exc}")
        return pd.DataFrame(), weights, dominant

    # Filter to ML-predicted top-2 emotions
    top2       = sorted(weights, key=weights.get, reverse=True)[:2]
    candidates = pool[pool["_pred_em"].isin(top2)].copy()
    if len(candidates) < n_songs:
        candidates = pool[pool["_pred_em"] == dominant].copy()
    if len(candidates) < n_songs:
        candidates = pool.copy()

    # Rank by Euclidean distance to target audio vector
    def audio_dist(row):
        d = 0.0
        for col in audio_cols:
            if col in row.index:
                v   = float(row[col]) if pd.notna(row.get(col)) else AUDIO_RANGES[col][2]
                tgt = target_audio.get(col, AUDIO_RANGES[col][2])
                if col == "Tempo":
                    v /= 200; tgt /= 200
                elif col == "Loudness":
                    v = (v + 60) / 60; tgt = (tgt + 60) / 60
                d += (v - tgt) ** 2
        return d ** 0.5

    candidates["_audio_dist"] = candidates.apply(audio_dist, axis=1)
    return candidates.nsmallest(n_songs, "_audio_dist").reset_index(drop=True), weights, dominant


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
bundle   = load_models()
metadata = load_metadata()
perf     = metadata.get("model_performance", {})

AVAILABLE_MODELS = [n for n in MODEL_FILES if bundle and n in bundle]
if not bundle or not AVAILABLE_MODELS:
    st.error(
        "No classifier models found in `model/`.\n\n"
        "Ensure `model/lightgbm_model.pkl`, `model/tfidf.pkl`, etc. are present."
    )
    st.stop()

# Load dataset from BigQuery
df = load_dataset_from_bq()
if df is None:
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

    st.markdown("#### 📊 Dataset (BigQuery)")
    st.metric("Tracks", f"{len(df):,}")
    ec = df["emotion_4"].value_counts()
    for em, cnt in ec.items():
        pct = cnt / len(df) * 100
        st.markdown(f"{EMOTION_STYLE[em]['emoji']} **{em.title()}** — {pct:.1f}%")
    st.caption(f"Source: `{BQ_TABLE_ID}`")

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
    if n_feat: st.metric("Feature Dimensions", f"{n_feat:,}")

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
    [
        f"{len(df):,}",
        f"{best_acc*100:.1f}%" if best_acc else "—",
        f"{best_f1:.4f}"       if best_f1  else "—",
        f"{n_feat_v:,}" if isinstance(n_feat_v, int) else str(n_feat_v),
    ],
    ["BQ Tracks Loaded", "Best Accuracy", "Best Weighted F1", "Feature Dimensions"],
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
    st.caption("Enter audio features and optional lyrics — the ML model predicts the emotion.")

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
        energy   = st.slider("⚡ Energy",           0.0, 1.0, 0.65, 0.01)
        dance    = st.slider("💃 Danceability",     0.0, 1.0, 0.60, 0.01)
        pos      = st.slider("😊 Positiveness",     0.0, 1.0, 0.50, 0.01)
        acoustic = st.slider("🎸 Acousticness",     0.0, 1.0, 0.30, 0.01)
        tempo    = st.slider("🥁 Tempo (BPM)",      60,  200, 120,  1)
        speech   = st.slider("🗣 Speechiness",      0.0, 1.0, 0.05, 0.01)
        instr    = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.10, 0.01)
        loudness = st.slider("🔊 Loudness (dB)",   -60.0, 0.0, -7.0, 0.5)
        pop      = st.slider("⭐ Popularity",       0,   100, 50,   1)

    st.markdown("---")
    if st.button("🎵 Classify Emotion", type="primary", use_container_width=True):
        song_vals = {
            "Energy": energy, "Danceability": dance, "Positiveness": pos,
            "Acousticness": acoustic, "Tempo": tempo, "Speechiness": speech,
            "Instrumentalness": instr, "Loudness": loudness,
            "Popularity": pop, "Liveness": 0.1,
        }
        with st.spinner("Running ML inference…"):
            X          = build_feature_matrix([song_vals], [lyrics], bundle)
            labels, pr = run_inference(X, bundle, model_key)

        em    = str(labels[0])
        pr0   = pr[0]
        le_   = bundle.get("label_encoder")
        cls   = list(le_.classes_) if le_ is not None else list(EMOTION_STYLE.keys())
        conf  = float(pr0.max())
        style = EMOTION_STYLE.get(em, {"color":"#64748b","emoji":"🎵","desc":""})
        probs = {str(c): float(p) for c, p in zip(cls, pr0)}

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
                color_discrete_map={
                    e.title(): EMOTION_STYLE.get(e, {}).get("color", "#ccc")
                    for e in probs
                },
                text="Probability",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(showlegend=False, height=320, yaxis_range=[0, 1],
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Similar songs from BigQuery dataset
        st.markdown("---")
        st.subheader(f"🎵 Similar {em.title()} Songs (from BigQuery)")
        sub       = df[df["emotion_4"] == em].copy()
        feat_cols = [c for c in ["Energy","Positiveness","Danceability","Acousticness","Tempo"]
                     if c in sub.columns]
        if not sub.empty and feat_cols:
            def _dist(row):
                d = 0.0
                for c in feat_cols:
                    v1 = 0.0 if pd.isna(row.get(c)) else float(row[c])
                    v2 = float(song_vals.get(c, 0.5))
                    if c == "Tempo": v1 /= 200; v2 /= 200
                    d += (v1 - v2) ** 2
                return d ** 0.5
            sub["_dist"] = sub.apply(_dist, axis=1)
            sim = (sub[sub["_dist"] > 0.01]
                   .nsmallest(10, "_dist")
                   [["song", "artist", "emotion_4", "Energy", "Positiveness", "_dist"]])
            sim = sim.rename(columns={
                "song":"🎵 Song","artist":"🎤 Artist","emotion_4":"🎭 Emotion",
                "Energy":"⚡ Energy","Positiveness":"😊 Positivity","_dist":"📏 Distance",
            })
            sim["📏 Distance"] = sim["📏 Distance"].round(4)
            st.dataframe(sim, use_container_width=True, hide_index=True)
        else:
            st.info("No similar songs found.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLORE DATASET  (live BigQuery)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🎵 Explore the Dataset")
    st.caption(f"Live from `{BQ_TABLE_ID}`")

    sel_em = st.selectbox(
        "Filter by emotion:",
        ["All"] + list(EMOTION_STYLE.keys()),
        format_func=lambda x: (
            f"{EMOTION_STYLE[x]['emoji']} {x.title()}" if x != "All" else "All Emotions"
        ),
    )

    fdf = df if sel_em == "All" else df[df["emotion_4"] == sel_em]
    st.info(f"Showing {len(fdf):,} songs")

    disp_cols  = [c for c in ["song","artist","emotion_4","Energy","Positiveness","Danceability"]
                  if c in fdf.columns]
    rename_map = {
        "song":"🎵 Song","artist":"🎤 Artist","emotion_4":"🎭 Emotion",
        "Energy":"⚡ Energy","Positiveness":"😊 Positivity","Danceability":"💃 Dance",
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
# TAB 3 — ANALYTICS  (live BigQuery)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Emotion Analytics")
    st.caption("All charts computed from live BigQuery data")

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
    avail_feat = [c for c in ["Energy","Positiveness","Danceability","Acousticness","Tempo"]
                  if c in df.columns]
    feat = st.selectbox("Feature:", avail_feat)
    fig2 = px.violin(
        df, x="emotion_4", y=feat, color="emotion_4",
        color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
        box=True, points=False, labels={"emotion_4":"Emotion", feat: feat},
    )
    fig2.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Feature Scatter by Emotion")
    avail2 = [c for c in ["Energy","Positiveness","Danceability","Acousticness"] if c in df.columns]
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
            {"Model":m, "Accuracy":v.get("accuracy",0), "Weighted F1":v.get("f1_weighted",0)}
            for m, v in perf.items()
        ]).sort_values("Accuracy", ascending=False)

        fig = px.bar(
            perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            title="Model Performance Comparison",
            color_discrete_sequence=["#1d4ed8","#3b82f6"],
        )
        fig.update_layout(height=420, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Breakdown")
        fb = metadata.get("feature_breakdown", {})
        if fb:
            fb_df  = pd.DataFrame({"Feature Group":list(fb.keys()), "Count":list(fb.values())})
            fb_fig = px.bar(fb_df, x="Feature Group", y="Count", color="Feature Group",
                            color_discrete_sequence=["#0f172a","#1d4ed8","#3b82f6"])
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
            Describe a vibe, a moment, a feeling. The pipeline reads your words via VADER + TF-IDF,
            maps them to emotion space, then runs the ML classifier on real BigQuery songs to
            curate your playlist. No rules — all signal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_pre, col_custom = st.columns([1, 2], gap="large")

    with col_pre:
        st.markdown("#### 🎛️ Pre-designed Vibes")
        st.caption("Click to generate a playlist from real BigQuery songs")
        for p in PRESET_PLAYLISTS:
            if st.button(
                f"{p['icon']}  {p['title']}\n{p['subtitle']}",
                key=f"preset_{p['id']}",
                use_container_width=True,
            ):
                st.session_state.update({
                    "pl_prompt": p["prompt"], "pl_n": p["n"],
                    "pl_title": f"{p['icon']} {p['title']}", "pl_run": True,
                })

    with col_custom:
        st.markdown("#### ✍️ Describe Your Own Vibe")
        st.caption("Type anything — a feeling, a scene, a moment.")
        prompt_input = st.text_area(
            "Your mood prompt",
            placeholder='e.g. "Sunday morning coffee, soft rain, feeling nostalgic but okay…"',
            height=110, label_visibility="collapsed", key="pl_prompt_input",
        )
        ca, cb, cc = st.columns([2, 1, 1])
        with ca:
            n_songs_custom = st.slider("Playlist length", 5, 40, 20, 5, key="pl_n_slider")
        with cb:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎵 Generate", type="primary",
                         use_container_width=True, key="pl_gen_btn"):
                if prompt_input.strip():
                    st.session_state.update({
                        "pl_prompt": prompt_input.strip(), "pl_n": n_songs_custom,
                        "pl_title": "🎧 Custom Playlist", "pl_run": True,
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
        with st.spinner("🔮 Classifying real BigQuery songs via ML model…"):
            pl_df, emo_weights, dominant = generate_playlist(
                prompt=active_prompt, df=df, bundle=bundle,
                model_key=model_key, n_songs=active_n,
            )
        st.session_state["pl_result"] = {
            "df": pl_df, "weights": emo_weights, "dominant": dominant,
            "title": active_title, "prompt": active_prompt,
        }

    # Render result
    res = st.session_state.get("pl_result")
    if res and not res["df"].empty:
        pl_df    = res["df"]
        weights  = res["weights"]
        dominant = res["dominant"]
        pl_title = res["title"]
        pl_prompt= res["prompt"]
        dom_s    = EMOTION_STYLE.get(dominant, {"color":"#1d4ed8","emoji":"🎵"})

        st.markdown("---")
        st.markdown(
            f'<div class="playlist-card" style="border-top:4px solid {dom_s["color"]};">'
            f'<div class="playlist-header">{pl_title}</div>'
            f'<div class="playlist-meta">Prompt: <em>"{pl_prompt}"</em> &nbsp;·&nbsp; '
            f'{len(pl_df)} real tracks from BigQuery &nbsp;·&nbsp; '
            f'Dominant: <strong>{dominant.title()}</strong> {dom_s["emoji"]}</div>'
            f'</div>', unsafe_allow_html=True,
        )

        rr1, rr2 = st.columns([1, 2])
        with rr1:
            st.markdown("**NLP Emotion Signal**")
            st.caption("VADER + TF-IDF cosine → emotion space")
            w_df = pd.DataFrame({
                "Emotion": [e.title() for e in weights],
                "Weight":  list(weights.values()),
            }).sort_values("Weight")
            fig_w = px.bar(
                w_df, x="Weight", y="Emotion", orientation="h", color="Emotion",
                color_discrete_map={e.title(): EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                text="Weight",
            )
            fig_w.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_w.update_layout(
                showlegend=False, height=220,
                xaxis_range=[0, max(weights.values())*1.35],
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=40,t=10,b=10),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        with rr2:
            st.markdown("**Playlist Audio Profile**")
            st.caption("Real BigQuery songs, coloured by ML-predicted emotion")
            color_col = "_pred_em" if "_pred_em" in pl_df.columns else "emotion_4"
            if "Energy" in pl_df.columns and "Positiveness" in pl_df.columns:
                hover = [c for c in ["song","artist"] if c in pl_df.columns]
                sc = px.scatter(
                    pl_df, x="Energy", y="Positiveness", color=color_col,
                    color_discrete_map={e: EMOTION_STYLE[e]["color"] for e in EMOTION_STYLE},
                    hover_data=hover, opacity=.85, height=220,
                )
                sc.update_layout(
                    showlegend=True, margin=dict(l=0,r=0,t=10,b=10),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(sc, use_container_width=True)

        st.markdown("---")
        st.markdown(f"### 🎵 Tracklist — {len(pl_df)} songs from BigQuery")

        conf_col = (f"_conf_{dominant}" if f"_conf_{dominant}" in pl_df.columns
                    else "_confidence" if "_confidence" in pl_df.columns else None)

        for idx, (_, row) in enumerate(pl_df.iterrows(), 1):
            em_r  = str(row.get("_pred_em", dominant))
            es    = EMOTION_STYLE.get(em_r, {"color":"#64748b","emoji":"🎵"})
            c_val = (float(row[conf_col])
                     if conf_col and pd.notna(row.get(conf_col)) else 0.0)
            c_pct = int(c_val * 100)
            st.markdown(
                f'<div class="track-row">'
                f'<span class="track-num">{idx}</span>'
                f'<div class="track-info">'
                f'<div class="track-name">{row.get("song","—")}</div>'
                f'<div class="track-artist">{row.get("artist","—")}</div>'
                f'</div>'
                f'<span class="track-badge" '
                f'style="background:{es["color"]}22;color:{es["color"]};">'
                f'{es["emoji"]} {em_r.title()}</span>'
                f'<div class="conf-wrap"><div class="conf-bg">'
                f'<div class="conf-fg" style="width:{c_pct}%;background:{es["color"]};"></div>'
                f'</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        export_cols = [c for c in ["song","artist","_pred_em","Energy","Positiveness",
                                    "Danceability","Tempo","_audio_dist","_confidence"]
                       if c in pl_df.columns]
        st.download_button(
            f"📥 Export Playlist CSV ({len(pl_df)} tracks)",
            pl_df[export_cols].rename(columns={
                "_pred_em":    "ML Predicted Emotion",
                "_confidence": "Model Confidence",
                "_audio_dist": "Audio Distance to Target",
            }).to_csv(index=False).encode(),
            "moodsense_playlist.csv", "text/csv",
            use_container_width=True,
        )

    elif st.session_state.get("pl_result") and res["df"].empty:
        st.warning("No songs matched. Try a different prompt or check models are loaded.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748b;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎶</div>
            <p style="font-size:1.1rem;font-weight:600;">Pick a preset or type your own mood above.</p>
            <p style="font-size:.9rem;">
                The ML model classifies real BigQuery songs and curates a playlist
                ranked by audio similarity to your emotional signal.
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
