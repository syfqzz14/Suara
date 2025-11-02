import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import tsfel
import joblib
import soundfile as sf
import tempfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# ==========================================================
# Konfigurasi halaman
# ==========================================================
st.set_page_config(
    page_title="Klasifikasi Suara Buka/Tutup",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Load model dan artefak
# ==========================================================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('artifacts/voice_classifier_model.pkl')
    scaler = joblib.load('artifacts/voice_scaler.pkl')
    feature_names = joblib.load('artifacts/feature_names.pkl')
    metadata = joblib.load('artifacts/model_metadata.pkl')
    cfg = tsfel.get_features_by_domain(["statistical"])
    return model, scaler, feature_names, metadata, cfg

# ==========================================================
# 🎚️ Preprocessing audio
# ==========================================================
def preprocess_audio(y, sr, target_sr=16000):
    if y is None or len(y) == 0:
        raise ValueError("File audio kosong atau tidak terbaca.")
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > 0 and np.max(np.abs(y_trimmed)) > 0:
        y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
    if len(y_trimmed) < sr * 0.1:
        raise ValueError("Audio terlalu pendek setelah trimming.")
    return y_trimmed, sr

# ==========================================================
# 🔍 Ekstraksi fitur gabungan (TSFEL + MFCC)
# ==========================================================
def extract_combined_features(y, sr, feature_names, cfg):
    try:
        if y is None or len(y) < 2048:
            raise ValueError(f"Audio terlalu pendek (len={len(y)})")
        X_tsfel = tsfel.time_series_features_extractor(cfg, y, fs=sr, verbose=0)
        if X_tsfel.shape[0] == 0:
            raise ValueError("Hasil ekstraksi TSFEL kosong.")
        tsfel_feats = X_tsfel.iloc[0].values
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_feats = np.concatenate([mfcc_mean, mfcc_std])
        all_features = np.concatenate([tsfel_feats, mfcc_feats])
        if len(all_features) != len(feature_names):
            diff = len(feature_names) - len(all_features)
            if diff > 0:
                all_features = np.append(all_features, np.zeros(diff))
            else:
                all_features = all_features[:len(feature_names)]
        return all_features.reshape(1, -1)
    except Exception as e:
        st.error(f"Error ekstraksi fitur: {e}")
        return None

# ==========================================================
# 🧠 Prediksi
# ==========================================================
def predict_audio(y, sr, model, scaler, feature_names, metadata, cfg):
    try:
        y_proc, sr_proc = preprocess_audio(y, sr, metadata["target_sr"])
        if len(y_proc) < 2048:
            st.error("Audio terlalu pendek setelah preprocessing.")
            return None, None, None
        features = extract_combined_features(y_proc, sr_proc, feature_names, cfg)
        if features is None:
            return None, None, None
        if np.isnan(features).any():
            features = np.nan_to_num(features)
        features_df = pd.DataFrame(features, columns=feature_names)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        return prediction, probabilities, y_proc
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None, None

# ==========================================================
# 🎧 Visualisasi audio
# ==========================================================
def plot_bar_and_spectrogram(y, sr):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plt.style.use('dark_background')

    # === Bar plot amplitudo ===
    step = max(1, len(y) // 200)
    time_axis = np.linspace(0, len(y) / sr, len(y))
    axes[0].bar(time_axis[::step], y[::step], width=0.002, color='#58a6ff', alpha=0.8)
    axes[0].set_title("📊 Amplitudo (Bar Plot)", fontsize=12, color='white')
    axes[0].set_xlabel("Waktu (detik)")
    axes[0].set_ylabel("Amplitudo")
    axes[0].grid(alpha=0.2, color='gray')

    # === Spektrogram ===
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis='time', y_axis='linear', ax=axes[1], cmap='magma')
    axes[1].set_title("🎶 Spectrogram (dB)", fontsize=12, color='white')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    return fig

# ==========================================================
# 🚀 Main App
# ==========================================================
def main():
    st.markdown("<h1 style='text-align:center;'>🎧 Klasifikasi Suara Buka/Tutup</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#cccccc;'>Deteksi otomatis jenis suara (buka/tutup) menggunakan model Machine Learning</p>", unsafe_allow_html=True)

    # 🌈 CSS gaya baru
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #1a1f2b, #0b0f16);
            color: #f1f1f1;
        }
        [data-testid="stSidebar"] {
            background-color: #0e1117;
            color: #ffffff;
            border-right: 2px solid #2a2a2a;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
        h1,h2,h3 {
            color: #00c6ff !important;
        }
        .audiobox {
            background: #161b22;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2a2a2a;
        }
    </style>
    """, unsafe_allow_html=True)

    model, scaler, feature_names, metadata, cfg = load_model_artifacts()

    # Sidebar
    with st.sidebar:
        st.header("Informasi Model")
        st.metric("Akurasi Training", f"{metadata['train_accuracy']:.2%}")
        st.metric("Akurasi Testing", f"{metadata['test_accuracy']:.2%}")
        st.metric("Jumlah Fitur", str(metadata['n_features']))
        st.metric("Sample Rate", f"{metadata['target_sr']} Hz")
        st.markdown("---")
        st.info("🎙️ Rekam atau upload file suara, lalu klik **Prediksi** untuk hasil klasifikasi.")

    st.subheader("Input Audio")
    mode = st.radio("Pilih sumber audio:", ["Rekam Langsung", "Upload File"])

    audio_data, sr = None, None
    if mode == "Rekam Langsung":
        audio_bytes = audio_recorder(text="", recording_color="#ff4b4b", neutral_color="#eeeeee", icon_size="2x")
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            st.markdown("<div class='audiobox'>", unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            st.markdown("</div>", unsafe_allow_html=True)
            audio_data = y
            os.remove(tmp_path)
    else:
        uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3"])
        if uploaded_file:
            y, sr = librosa.load(uploaded_file, sr=None, mono=True)
            st.markdown("<div class='audiobox'>", unsafe_allow_html=True)
            st.audio(uploaded_file, format="audio/wav")
            st.markdown("</div>", unsafe_allow_html=True)
            audio_data = y

    # Tombol prediksi
    if audio_data is not None and sr is not None:
        if st.button("Prediksi", use_container_width=True):
            with st.spinner("Memproses dan memprediksi..."):
                pred, prob, y_proc = predict_audio(audio_data, sr, model, scaler, feature_names, metadata, cfg)
                if pred is not None:
                    label = metadata["label_map"][pred]
                    st.success(f"Hasil Prediksi: **{label.upper()}**")

                    # Diagram batang probabilitas
                    st.markdown("###Probabilitas Klasifikasi")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    labels = list(metadata["label_map"].values())
                    probs = [prob[i]*100 for i in range(len(labels))]
                    ax.bar(labels, probs, color=['#00c6ff', '#ff6f61'], alpha=0.8)
                    ax.set_ylabel("Persentase (%)")
                    ax.set_ylim(0, 100)
                    for i, v in enumerate(probs):
                        ax.text(i, v + 2, f"{v:.1f}%", ha='center', color='white', fontsize=10)
                    st.pyplot(fig)

                    # Visualisasi audio
                    st.markdown("###Visualisasi Audio")
                    st.pyplot(plot_bar_and_spectrogram(y_proc, metadata["target_sr"]))

if __name__ == "__main__":
    main()
