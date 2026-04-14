"""
Speech Emotion Recognition Dashboard
Upload audio or record live, get emotion predictions and song recommendations.
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import joblib
import json
import os
import tempfile
import warnings
from datetime import datetime
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

from song_recommender import get_recommendations

# Constants
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
MODELS_DIR = "models"

EMOTIONS_INFO = {
    'angry':   {'color': '#DC143C', 'desc': 'Loud, harsh, high energy'},
    'boredom': {'color': '#A9A9A9', 'desc': 'Monotonous, low energy'},
    'calm':    {'color': '#32CD32', 'desc': 'Smooth, relaxed pace'},
    'disgust': {'color': '#228B22', 'desc': 'Nasal, contemptuous tone'},
    'fear':    {'color': '#8B4513', 'desc': 'Trembling, high pitch'},
    'happy':   {'color': '#FFD700', 'desc': 'Positive, cheerful tone'},
    'neutral': {'color': '#808080', 'desc': 'Flat, monotone speech'},
    'ps':      {'color': '#FF69B4', 'desc': 'Sudden pitch changes'},
    'sad':     {'color': '#4169E1', 'desc': 'Low pitch, slow tempo'},
}


def extract_features(audio, sr=SAMPLE_RATE):
    """Same 218 features as the training notebook."""
    try:
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        features = []

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs.T, axis=0))
        features.extend(np.std(mfccs.T, axis=0))

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        features.extend(np.mean(mel_spec.T, axis=0)[:20])
        features.extend(np.std(mel_spec.T, axis=0)[:20])

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features.extend(np.mean(contrast.T, axis=0))
        features.extend(np.std(contrast.T, axis=0))

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features.extend(np.mean(tonnetz.T, axis=0))
        features.extend(np.std(tonnetz.T, axis=0))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
            pitch_vals = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 0:
                    pitch_vals.append(p)
            if len(pitch_vals) > 0:
                pv = np.array(pitch_vals)
                features.append(np.mean(pv))
                features.append(np.std(pv))
                features.append(np.median(pv))
                features.append(np.max(pv) - np.min(pv))
                pd_diff = np.diff(pv)
                features.append(np.mean(np.abs(pd_diff)))
                features.append(np.std(pd_diff))
            else:
                features.extend([0] * 6)
        except:
            features.extend([0] * 6)

        try:
            y_harm, y_perc = librosa.effects.hpss(audio)
            features.append(np.sum(np.abs(y_harm)) / (np.sum(np.abs(audio)) + 1e-10))
            h_e = np.sum(y_harm**2)
            p_e = np.sum(y_perc**2)
            features.append(h_e / (h_e + p_e + 1e-10))
        except:
            features.extend([0] * 2)

        try:
            fft_mag = np.abs(np.fft.fft(audio)[:len(audio)//2])
            freqs = np.fft.fftfreq(len(audio), 1/sr)[:len(fft_mag)]
            peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag)*0.1, distance=100)
            if len(peaks) > 0:
                pf = freqs[peaks]
                for i in range(min(3, len(pf))):
                    features.append(pf[i])
                features.extend([0] * max(0, 3 - len(pf)))
            else:
                features.extend([0] * 3)
        except:
            features.extend([0] * 3)

        try:
            frames_e = []
            for i in range(0, len(audio) - 2048, 512):
                frames_e.append(np.sum(audio[i:i+2048]**2))
            if len(frames_e) > 1:
                fe = np.array(frames_e)
                features.append(np.std(fe) / (np.mean(fe) + 1e-10))
                features.append(np.polyfit(range(len(fe)), fe, 1)[0] if len(fe) > 2 else 0)
            else:
                features.extend([0] * 2)
        except:
            features.extend([0] * 2)

        try:
            stft_mag = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
            flux = np.sum(np.diff(stft_mag, axis=1)**2, axis=0)
            features.append(np.mean(flux))
            features.append(np.std(flux))
        except:
            features.extend([0] * 2)

        try:
            zcr_f = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
            features.append(np.std(zcr_f) / (np.mean(zcr_f) + 1e-10) if len(zcr_f) > 1 else 0)
            rms_f = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            features.append(np.std(rms_f) / (np.mean(rms_f) + 1e-10) if len(rms_f) > 1 else 0)
        except:
            features.extend([0] * 2)

        try:
            cf = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.append(np.std(cf) / (np.mean(cf) + 1e-10) if len(cf) > 1 else 0)
        except:
            features.append(0)

        try:
            onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
            if len(onsets) > 0:
                os_str = librosa.onset.onset_strength(y=audio, sr=sr)
                features.append(np.mean(os_str))
                features.append(np.max(os_str))
                features.append(len(onsets) / DURATION)
                if len(onsets) > 1:
                    oi = np.diff(onsets)
                    features.append(np.std(oi) / (np.mean(oi) + 1e-10))
                else:
                    features.append(0)
            else:
                features.extend([0] * 4)
        except:
            features.extend([0] * 4)

        try:
            stft_m = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
            fr = librosa.fft_frequencies(sr=sr, n_fft=len(stft_m))
            lo = np.mean(np.sum(stft_m[(fr >= 0) & (fr < 500), :], axis=0))
            mi = np.mean(np.sum(stft_m[(fr >= 500) & (fr < 2000), :], axis=0))
            hi = np.mean(np.sum(stft_m[(fr >= 2000) & (fr < 8000), :], axis=0))
            tot = lo + mi + hi + 1e-10
            features.extend([lo/tot, mi/tot, hi/tot])
            features.append(np.std([lo, mi, hi]) / (np.mean([lo, mi, hi]) + 1e-10))
        except:
            features.extend([0] * 4)

        try:
            stft_m = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
            fr = librosa.fft_frequencies(sr=sr, n_fft=len(stft_m))
            env = np.mean(stft_m, axis=1)
            vf = fr[:len(env)]
            vf = vf[vf > 0]
            ve = env[:len(vf)]
            if len(vf) > 1 and np.sum(ve) > 0:
                features.append(np.polyfit(np.log10(vf+1), np.log10(ve+1e-10), 1)[0])
            else:
                features.append(0)
        except:
            features.append(0)

        try:
            rms_r = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            thr = np.percentile(rms_r, 30)
            features.append(np.sum(rms_r > thr) / len(rms_r))
            if len(rms_r) > 2:
                pk, _ = find_peaks(rms_r, height=thr, distance=5)
                if len(pk) > 1:
                    pi = np.diff(pk)
                    features.append(1.0 / (np.std(pi)/(np.mean(pi)+1e-10) + 1e-10))
                else:
                    features.append(0)
            else:
                features.append(0)
        except:
            features.extend([0] * 2)

        try:
            stft_m = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
            ss = [1.0/(np.std(stft_m[:,t])/(np.mean(stft_m[:,t])+1e-10)+1e-10)
                  for t in range(stft_m.shape[1])]
            features.append(np.mean(ss))
            features.append(np.std(ss))
        except:
            features.extend([0] * 2)

        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
            pv2 = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 0:
                    pv2.append(p)
            if len(pv2) > 2:
                pv2 = np.array(pv2)
                features.append(np.polyfit(np.arange(len(pv2)), pv2, 1)[0])
                features.append(1.0/(np.std(pv2)/(np.mean(pv2)+1e-10)+1e-10))
            else:
                features.extend([0] * 2)
        except:
            features.extend([0] * 2)

        try:
            rms_ad = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            if len(rms_ad) > 10:
                af = int(len(rms_ad)*0.2)
                df = int(len(rms_ad)*0.2)
                ae = np.mean(rms_ad[:af])
                de = np.mean(rms_ad[-df:])
                se = np.mean(rms_ad[af:-df]) if af+df < len(rms_ad) else np.mean(rms_ad)
                t_e = ae + se + de + 1e-10
                features.extend([ae/t_e, se/t_e, de/t_e])
                features.append((rms_ad[af-1]-rms_ad[0])/af if af > 1 else 0)
            else:
                features.extend([0] * 4)
        except:
            features.extend([0] * 4)

        try:
            stft_m = np.abs(librosa.stft(audio, hop_length=HOP_LENGTH))
            fr = librosa.fft_frequencies(sr=sr, n_fft=len(stft_m))
            fb = []
            for t in range(min(10, stft_m.shape[1])):
                sp = stft_m[:, t]
                pk, _ = find_peaks(sp, height=np.max(sp)*0.2, distance=50)
                if len(pk) >= 2:
                    for peak in pk[:2]:
                        hp = sp[peak]/np.sqrt(2)
                        li = np.where(sp[:peak] >= hp)[0]
                        ri = np.where(sp[peak:] >= hp)[0]
                        if len(li) > 0 and len(ri) > 0:
                            fb.append(fr[peak+ri[-1]] - fr[li[0]])
            features.append(np.mean(fb) if fb else 0)
        except:
            features.append(0)

        return np.array(features)
    except Exception as e:
        return None


def load_model_by_name(model_name):
    for ext in ['.joblib', '.pkl']:
        path = os.path.join(MODELS_DIR, f'{model_name.lower()}{ext}')
        if os.path.exists(path):
            return joblib.load(path)
    return None


def load_scaler_and_encoder():
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    return scaler, le


def predict_emotion(audio, sr, model, scaler, le):
    feat = extract_features(audio, sr)
    if feat is None:
        return None, None
    expected = scaler.n_features_in_
    if len(feat) < expected:
        feat = np.pad(feat, (0, expected - len(feat)))
    elif len(feat) > expected:
        feat = feat[:expected]
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    probs = model.predict_proba(feat_scaled)[0]
    emotions = le.classes_
    results = {emo: float(p) for emo, p in zip(emotions, probs)}
    dominant = max(results, key=results.get)
    return results, dominant


def show_results(results, dominant, audio_data, sr, audio_source, model_option):
    #Display all analysis results.
    st.markdown("Analysis Results")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        emotions = list(results.keys())
        probs = list(results.values())
        colors = [EMOTIONS_INFO.get(e, {}).get('color', '#808080') for e in emotions]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(emotions, probs, color=colors)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, max(probs) * 1.3)
        ax.set_title(f"Detected: {dominant.upper()} ({results[dominant]*100:.1f}%)", fontweight='bold', fontsize=14)
        for b, p in zip(bars, probs):
            if p > 0.03:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                        f'{p*100:.1f}%', ha='center', fontsize=8)
        plt.xticks(rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("Detailed Probabilities")
        for emo in emotions:
            prob = results[emo]
            emoji = EMOTIONS_INFO.get(emo, {}).get('emotion')
            st.progress(float(prob), text=f"{emoji} {emo.capitalize()}: {prob*100:.1f}%")

    with col_side:
        emoji = EMOTIONS_INFO.get(dominant, {}).get('emotion')
        color = EMOTIONS_INFO.get(dominant, {}).get('color', '#808080')
        desc = EMOTIONS_INFO.get(dominant, {}).get('desc', '')
        st.markdown(f"""
        <div style="border: 3px solid {color}; border-radius: 15px; padding: 25px; text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 4em; margin: 0;">{emoji}</h1>
            <h2 style="color: {color}; margin: 10px 0;">{dominant.upper()}</h2>
            <h3>{results[dominant]*100:.1f}%</h3>
            <p style="color: #666;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Analysis Info")
        st.write(f"**Source:** {audio_source}")
        st.write(f"**Model:** {model_option}")
        st.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

        st.markdown("#### Top 3 Emotions")
        sorted_probs = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
        for emo, prob in sorted_probs:
            e = EMOTIONS_INFO.get(emo, {}).get('emotion')
            st.write(f"{e} **{emo.capitalize()}**: {prob*100:.1f}%")

    # Waveform
    st.markdown("Audio Waveform")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    t = np.linspace(0, len(audio_data)/sr, len(audio_data))
    ax2.plot(t, audio_data, color='steelblue', linewidth=0.5)
    ax2.fill_between(t, audio_data, alpha=0.3, color='steelblue')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Audio Waveform")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

    # Song recommendations
    st.markdown(f"Song Recommendations")
    st.markdown(f"Based on your detected emotion **{dominant.upper()}**, here are songs to help with your mood:")
    recs = get_recommendations(dominant, num_recommendations=5)
    if recs:
        for i, song in enumerate(recs, 1):
            col_s, col_m = st.columns([4, 1])
            with col_s:
                st.write(f"**{i}. {song['title']}** — {song['artist']} ({song['genre']})")
            with col_m:
                st.write(f"Match: {song['similarity_score']:.1%}")

    # Save to history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'Time': datetime.now().strftime("%H:%M:%S"),
        'Emotion': f"{EMOTIONS_INFO.get(dominant, {}).get('emoji', '')} {dominant.capitalize()}",
        'Confidence': f"{results[dominant]*100:.1f}%",
        'Model': model_option,
        'Source': audio_source
    })


#PAGE CONFIG
st.set_page_config(page_title="SER Dashboard", page_icon="🎤", layout="wide", initial_sidebar_state="expanded")

#SESSION STATE INIT
if 'history' not in st.session_state:
    st.session_state.history = []
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'recorded_source' not in st.session_state:
    st.session_state.recorded_source = None
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None

#SIDEBAR
with st.sidebar:
    st.markdown("Model Selection")
    model_option = st.selectbox("Select Model", ["MLP", "SVM", "Random_Forest", "Gradient_Boosting"], index=0)
    st.markdown("Input Settings")
    input_mode = st.radio("Input Mode", ["Upload Audio File", "Record Live"], index=0)

    if input_mode == "Record Live":
        duration = st.slider("Recording Duration (seconds)", 2, 10, 3)
        sample_rate_option = st.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
    else:
        duration = 3
        sample_rate_option = 22050

    st.markdown("Emotions Detected")
    st.markdown("""
     Angry &nbsp;&nbsp;  Boredom &nbsp;&nbsp;  Calm  
     Disgust &nbsp;&nbsp;  Fear &nbsp;&nbsp;  Happy  
     Neutral &nbsp;&nbsp;  Surprise &nbsp;&nbsp;  Sad
    """)


    st.markdown("Model Status")
    model_found = os.path.exists(os.path.join(MODELS_DIR, f'{model_option.lower()}.joblib')) or \
                  os.path.exists(os.path.join(MODELS_DIR, f'{model_option.lower()}.pkl'))
    if model_found:
        st.success(f"{model_option} model found")
    else:
        st.warning(f"{model_option} model not found")

    metadata = {}
    if os.path.exists(os.path.join(MODELS_DIR, 'metadata.json')):
        with open(os.path.join(MODELS_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        st.markdown("### Model Info")
        st.write(f"Best model: **{metadata.get('best_model', 'N/A')}**")
        st.write(f"Accuracy: **{metadata.get('test_accuracy', 0)*100:.1f}%**")
        st.write(f"Emotions: **{metadata.get('n_classes', 0)}**")
        st.write(f"Features: **{metadata.get('n_features', 0)}**")

#LOAD MODEL
try:
    model = load_model_by_name(model_option)
    scaler, le = load_scaler_and_encoder()
    if model is None:
        st.error(f"{model_option} model not found. Run the notebook first.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

#HEADER
st.title("🎤 Speech Emotion Recognition Dashboard")
st.markdown("Detect emotions from speech and get personalized song recommendations to uplift your mood.")

#AUDIO INPUT
st.markdown("Audio Input")

audio_data = None
sr = SAMPLE_RATE
audio_source = ""

if input_mode == "Upload Audio File":
    # Clear any previous recording
    st.session_state.recorded_audio = None

    uploaded = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.write(uploaded.getvalue())
        tmp.close()
        st.audio(uploaded)
        audio_data, sr = librosa.load(tmp.name, sr=SAMPLE_RATE, duration=DURATION)
        os.unlink(tmp.name)
        audio_source = "File Upload"

        # Auto analyze
        with st.spinner("Analyzing..."):
            results, dominant = predict_emotion(audio_data, sr, model, scaler, le)
        if results and dominant:
            show_results(results, dominant, audio_data, sr, audio_source, model_option)
        else:
            st.error("Could not analyze audio. Try a different file.")

elif input_mode == "Record Live":
    # Show record button
    if st.button("Start Recording", type="primary", use_container_width=True):
        try:
            import sounddevice as sd
            import soundfile as sf

            with st.spinner(f"Recording for {duration} seconds."):
                recording = sd.rec(int(duration * sample_rate_option),
                                   samplerate=sample_rate_option, channels=1, dtype='float32')
                sd.wait()

            audio_recorded = recording.flatten()

            # Check if we got actual audio
            if np.max(np.abs(audio_recorded)) < 0.001:
                st.warning("No audio detected. Check your microphone.")
            else:
                # Resample if needed
                if sample_rate_option != SAMPLE_RATE:
                    audio_recorded = librosa.resample(audio_recorded, orig_sr=sample_rate_option, target_sr=SAMPLE_RATE)

                # Normalize
                max_val = np.max(np.abs(audio_recorded))
                if max_val > 0:
                    audio_recorded = audio_recorded / max_val

                # Save to session state
                st.session_state.recorded_audio = audio_recorded
                st.session_state.recorded_source = "Live Recording"
                st.rerun()

        except ImportError:
            st.error("sounddevice not installed. Run: pip install sounddevice soundfile")
        except Exception as e:
            st.error(f"Recording failed: {e}")
            st.info("Check microphone permissions in System Settings > Privacy > Microphone.")

    # If we have a recording from session state, analyze it
    if st.session_state.recorded_audio is not None:
        audio_data = st.session_state.recorded_audio
        sr = SAMPLE_RATE
        audio_source = st.session_state.recorded_source or "Live Recording"

        # Play back the recording
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        import soundfile as sf
        sf.write(tmp.name, audio_data, SAMPLE_RATE)
        st.audio(tmp.name)
        os.unlink(tmp.name)

        st.success("Recording loaded! Analyzing...")

        # Auto analyze
        with st.spinner("Analyzing."):
            results, dominant = predict_emotion(audio_data, sr, model, scaler, le)

        if results and dominant:
            show_results(results, dominant, audio_data, sr, audio_source, model_option)
        else:
            st.error("Could not analyze audio. Try recording again.")

        # Clear button
        if st.button("Clear Recording & Record Again"):
            st.session_state.recorded_audio = None
            st.session_state.recorded_source = None
            st.rerun()

#HISTORY
if st.session_state.history:
    st.markdown("Analysis History")
    import pandas as pd
    df = pd.DataFrame(reversed(st.session_state.history[-10:]))
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

#FOOTER
st.markdown("""
<div style="text-align: center; color: gray; padding: 10px;">
    Speech Emotion Recognition Dashboard | 9 Emotions | RAVDESS + TESS + EMO-DB
</div>
""", unsafe_allow_html=True)