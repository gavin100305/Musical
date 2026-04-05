import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

from transformers import AutoProcessor, MusicgenForConditionalGeneration

st.set_page_config(
    page_title="MusicGen Studio",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -- Theme colors (Dark Mode Only) ------------------------------------------
BG          = "#0a0a0a"
SURFACE     = "#141414"
SURFACE2    = "#111111"
BORDER      = "#2a2a2a"
TEXT        = "#f0f0f0"
TEXT_MUTED  = "#97C459"
TEXT_LABEL  = "#c0dd97"
PRIMARY     = "#97C459"
PRIMARY_TXT = "#0a0a0a"
ACCENT_BG   = "#141414"
PILL_BG     = "#1a1a1a"
PILL_TXT    = "#97C459"
CARD_BG     = "#111111"
INPUT_BG    = "#141414"
INPUT_BDR   = "#2a2a2a"
DIVIDER     = "#1f1f1f"
TAG_BG      = "#1a1a1a"
TAG_TXT     = "#97C459"
BTN_HOVER   = "#639922"
DL_TXT      = "#97C459"
DL_BDR      = "#639922"
DL_HOVER    = "#1a1a1a"
METRIC_BG   = "#141414"
METRIC_LBL  = "#97C459"
METRIC_VAL  = "#f0f0f0"
EXPANDER_BG = "#111111"
EXPANDER_BD = "#2a2a2a"

# -- Inject CSS ----------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, *, [class*="st-"] {{
        font-family: 'Poppins', sans-serif !important;
    }}

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {{
        background-color: {BG} !important;
    }}

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    footer {{
        display: none !important;
    }}

    .block-container {{
        max-width: 740px !important;
        padding: 3rem 2rem 5rem 2rem !important;
    }}

    .page-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: {TEXT_MUTED};
        margin-bottom: 0.35rem;
    }}

    .page-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }}

    .page-sub {{
        font-size: 0.88rem;
        color: {TEXT_MUTED};
        margin-bottom: 0;
        font-weight: 400;
    }}

    .divider {{
        border: none;
        border-top: 1px solid {DIVIDER};
        margin: 1.8rem 0;
    }}

    .section-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {TEXT_LABEL};
        margin-bottom: 0.5rem;
    }}

    textarea {{
        background-color: {INPUT_BG} !important;
        border: 1.5px solid {INPUT_BDR} !important;
        border-radius: 14px !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.93rem !important;
        color: {TEXT} !important;
        padding: 14px 16px !important;
        line-height: 1.6 !important;
    }}

    textarea:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px {PRIMARY}22 !important;
        outline: none !important;
    }}

    textarea::placeholder {{
        color: {TEXT_MUTED} !important;
        opacity: 0.6 !important;
    }}

    [data-testid="stSelectbox"] > div > div {{
        background-color: {INPUT_BG} !important;
        border: 1.5px solid {INPUT_BDR} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.88rem !important;
    }}

    label, [data-testid="stWidgetLabel"] p {{
        color: {TEXT_LABEL} !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        letter-spacing: 0.02em;
    }}

    .stButton > button {{
        background-color: {PRIMARY} !important;
        color: {PRIMARY_TXT} !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 0.93rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        letter-spacing: 0.02em;
        width: 100%;
        transition: background-color 0.2s ease;
    }}

    .stButton > button:hover {{
        background-color: {BTN_HOVER} !important;
    }}

    .stDownloadButton > button {{
        background-color: transparent !important;
        color: {DL_TXT} !important;
        border: 1.5px solid {DL_BDR} !important;
        border-radius: 10px !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
    }}

    .stDownloadButton > button:hover {{
        background-color: {DL_HOVER} !important;
    }}

    [data-testid="stMetric"] {{
        background-color: {METRIC_BG} !important;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid {BORDER};
    }}

    [data-testid="stMetricLabel"] p {{
        color: {METRIC_LBL} !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    [data-testid="stMetricValue"] {{
        color: {METRIC_VAL} !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.88rem !important;
    }}

    [data-testid="stExpander"] {{
        background-color: {EXPANDER_BG} !important;
        border: 1px solid {EXPANDER_BD} !important;
        border-radius: 14px !important;
    }}

    [data-testid="stExpander"] summary p {{
        font-size: 0.86rem !important;
        color: {TEXT_LABEL} !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
    }}

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] strong {{
        color: {TEXT} !important;
        font-size: 0.88rem !important;
    }}

    /* Fix status bar theme */
    [data-testid="stStatusWidget"] {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
    }}

    [data-testid="stStatusWidget"] span,
    [data-testid="stStatusWidget"] p,
    [data-testid="stStatusWidget"] div {{
        color: {TEXT} !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.85rem !important;
    }}

    /* Remove red border on textarea completely */
    textarea {{
        outline: none !important;
        box-shadow: none !important;
    }}

    textarea:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 2px {PRIMARY}33 !important;
        outline: none !important;
    }}

    /* Kill Streamlit's internal red validation border */
    [data-baseweb="textarea"] {{
        border: none !important;
        box-shadow: none !important;
    }}

    [data-baseweb="textarea"]:focus-within {{
        border: none !important;
        box-shadow: none !important;
    }}

    div[data-baseweb="textarea"] > div {{
        border-color: {INPUT_BDR} !important;
        box-shadow: none !important;
        background-color: {INPUT_BG} !important;
    }}

    div[data-baseweb="textarea"] > div:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 2px {PRIMARY}33 !important;
    }}

    audio {{
        width: 100%;
        border-radius: 10px;
        margin-top: 0.4rem;
    }}

    .prompt-bar {{
        background-color: {ACCENT_BG};
        border-left: 3px solid {PRIMARY};
        border-radius: 0 10px 10px 0;
        padding: 0.65rem 1rem;
        margin: 0.8rem 0 1.2rem 0;
        font-size: 0.86rem;
        color: {TEXT};
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }}

    .clip-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
    }}

    .clip-tag {{
        display: inline-block;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {TAG_TXT};
        background-color: {TAG_BG};
        border-radius: 6px;
        padding: 2px 9px;
        margin-bottom: 0.45rem;
    }}

    .clip-text {{
        font-size: 0.92rem;
        color: {TEXT};
        font-weight: 500;
        line-height: 1.5;
    }}

    .loading-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        margin: 1rem 0;
    }}

    .loading-title {{
        font-weight: 600;
        font-size: 1rem;
        color: {TEXT};
        margin-bottom: 0.3rem;
    }}

    .loading-sub {{
        font-size: 0.82rem;
        color: {TEXT_MUTED};
    }}

    .info-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 1rem;
        margin-bottom: 1.4rem;
    }}

    .info-pill {{
        font-size: 0.73rem;
        font-weight: 500;
        color: {PILL_TXT};
        background-color: {PILL_BG};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 3px 12px;
        font-family: 'Poppins', sans-serif;
    }}

</style>
""", unsafe_allow_html=True)

# -- Top bar: title --------- ------------------------------------------------
st.markdown('<p class="page-label">LLM & Gen AI Lab — Experiment 6 — Task 3</p>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">MusicGen Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Generate instrumental music from a text description using AI</p>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -- Model loading -------------------------------------------------------------
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return processor, model

if "model_loaded" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
    <div class="loading-card">
        <div class="loading-title">Loading MusicGen model</div>
        <div class="loading-sub">This only happens once. Should take 1 to 2 minutes.</div>
    </div>
    """, unsafe_allow_html=True)
    processor, model = load_model()
    st.session_state.model_loaded = True
    loading_placeholder.empty()
else:
    processor, model = load_model()

# -- Helpers -------------------------------------------------------------------
def build_prompt(user_text, genre, mood):
    parts = [user_text.strip()]
    if genre != "None":
        parts.append(genre.lower() + " music")
    if mood != "None":
        parts.append(mood.lower())
    return ", ".join(parts)

def generate_music(prompt, duration):
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    max_new_tokens = int(duration * 51.2)
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_data = audio_values[0, 0].cpu().numpy()
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16, sampling_rate

# -- Session state -------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -- Prompt input --------------------------------------------------------------
st.markdown('<p class="section-label">Describe your music</p>', unsafe_allow_html=True)

user_prompt = st.text_area(
    "",
    placeholder='e.g. "a calm piano melody with soft strings in the background"',
    height=110,
    label_visibility="collapsed"
)

# -- Settings ------------------------------------------------------------------
st.markdown('<p class="section-label" style="margin-top:1.2rem;">Settings</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    genre = st.selectbox(
        "Genre",
        ["None", "Classical", "Jazz", "Ambient", "Electronic", "Cinematic", "Lo-fi"]
    )

with col2:
    mood = st.selectbox(
        "Mood",
        ["None", "Calm", "Joyful", "Suspenseful", "Sad", "Energetic", "Romantic"]
    )

with col3:
    duration_seconds = st.slider(
        "Duration (seconds)",
        min_value=5, max_value=30, value=10, step=5
    )

# -- Prompt preview ------------------------------------------------------------
if user_prompt.strip():
    enriched = build_prompt(user_prompt, genre, mood)
    st.markdown(
        f'<div class="prompt-bar">Sending to model: <strong>{enriched}</strong></div>',
        unsafe_allow_html=True
    )

# -- Info pills ----------------------------------------------------------------
st.markdown(f"""
<div class="info-row">
    <span class="info-pill">facebook/musicgen-small</span>
    <span class="info-pill">HuggingFace Transformers</span>
    <span class="info-pill">WAV 32kHz mono</span>
</div>
""", unsafe_allow_html=True)

# -- Generate button -----------------------------------------------------------
generate_btn = st.button("Generate Music", use_container_width=True)

if generate_btn:
    if not user_prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        enriched_prompt = build_prompt(user_prompt, genre, mood)
        with st.spinner(f"Generating {duration_seconds}s of music... this may take 20 to 60 seconds on CPU."):
            start = time.time()
            try:
                audio_data, sample_rate = generate_music(enriched_prompt, duration_seconds)
                elapsed = round(time.time() - start, 1)

                os.makedirs("outputs", exist_ok=True)
                filename = f"outputs/music_{int(time.time())}.wav"
                wavfile.write(filename, sample_rate, audio_data)

                st.session_state.history.append({
                    "prompt": enriched_prompt,
                    "file": filename,
                    "duration": duration_seconds,
                    "elapsed": elapsed,
                    "genre": genre,
                    "mood": mood,
                })
                st.success(f"Done. Generated in {elapsed}s.")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

# -- Results -------------------------------------------------------------------
if st.session_state.history:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Generated clips</p>', unsafe_allow_html=True)

    for i, entry in enumerate(reversed(st.session_state.history)):
        clip_num = len(st.session_state.history) - i
        prompt_display = entry['prompt'][:90] + "..." if len(entry['prompt']) > 90 else entry['prompt']

        st.markdown(f"""
        <div class="clip-card">
            <div class="clip-tag">Clip {clip_num}</div>
            <div class="clip-text">{prompt_display}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Duration", f"{entry['duration']}s")
        m2.metric("Genre", entry['genre'])
        m3.metric("Generated in", f"{entry['elapsed']}s")

        with open(entry['file'], "rb") as f:
            audio_bytes = f.read()

        st.audio(audio_bytes, format="audio/wav")

        st.download_button(
            label="Download WAV",
            data=audio_bytes,
            file_name=os.path.basename(entry['file']),
            mime="audio/wav",
            key=f"dl_{i}"
        )

        if i < len(st.session_state.history) - 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -- Observations --------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with st.expander("Observations and Notes"):
    st.markdown(f"""
<span style='color:{TEXT}'>

**Model:** facebook/musicgen-small (300M parameters)  
**Audio format:** WAV, 32kHz mono

**Report notes:**
- Specific instrument names like piano, guitar, or violin produce more targeted results
- Mood keywords like calm or energetic noticeably affect tempo and energy
- CPU generation takes 20 to 60 seconds per 10s clip; a T4 GPU reduces this to around 5 seconds
- guidance scale of 3.0 controls prompt adherence — higher values produce more literal output

**Limitations:**
- musicgen-small can produce repetitive patterns on longer durations
- No vocal generation — purely instrumental output
- Output quality varies across different prompts

</span>
""", unsafe_allow_html=True)