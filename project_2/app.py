import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import librosa
import io
import joblib
from scipy import stats

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Sistem Identifikasi Suara - Speaker Verification",
    page_icon="🎤",
    layout="wide"
)

# ===============================
# Load Model, Scaler, Label Encoder, dan Metadata - SPEAKER VERIFICATION
# ===============================

def find_model_files():
    """
    Cari file model di berbagai lokasi yang mungkin
    Returns: models_dir (direktori yang berisi file model)
    """
    # Dapatkan direktori file saat ini
    if '__file__' in globals():
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        current_dir = os.getcwd()
    
    # Daftar lokasi yang mungkin
    possible_dirs = [
        os.path.join(os.path.dirname(current_dir), "tugas", "models2"),  # ../tugas/models2/ (NEW!)
        os.path.join(current_dir, "models2"),  # project_2/models2/
        os.path.join(current_dir, "models"),   # project_2/models/
        current_dir,                           # project_2/
        os.path.join(os.path.dirname(current_dir), "tugas", "models"),  # ../tugas/models/
        "/workspaces/PSD/tugas/models2",       # Absolute path models2 (development)
        "/workspaces/PSD/project_2/models",    # Absolute path (development)
        "/workspaces/PSD/tugas/models",        # Absolute path tugas (development)
        "/mount/src/psd/tugas/models2",        # Streamlit Cloud path
        "/mount/src/psd/project_2/models",     # Streamlit Cloud path
        "/mount/src/psd/models",               # Streamlit Cloud alternative
    ]
    
    # Cek setiap direktori
    for directory in possible_dirs:
        if os.path.exists(directory):
            # Cek speaker_model.pkl (model baru dengan speaker verification)
            speaker_model_path = os.path.join(directory, "speaker_model.pkl")
            if os.path.exists(speaker_model_path):
                return directory
    
    return None

# Cari direktori model
models_dir = find_model_files()

if models_dir is None:
    st.error("❌ File model tidak ditemukan!")
    st.warning("""
    **📁 Lokasi yang sudah dicek:**
    - `project_2/models/`
    - `project_2/`
    - `../tugas/models/`
    
    **🔧 Solusi:**
    
    **Opsi 1: Copy dari tugas/models/**
    ```bash
    cd /workspaces/PSD
    cp tugas/models/speaker_model*.pkl project_2/models/
    ```
    
    **Opsi 2: Training model dulu**
    1. Buka notebook: `tugas/Identifikasi_Suara_Buka_Tutup.ipynb`
    2. Jalankan semua cells
    3. File .pkl akan tersimpan di `tugas/models/`
    4. Copy ke `project_2/models/`
    
    **Opsi 3: Upload manual**
    - Upload 5 file .pkl ke folder `project_2/models/`
    - Restart aplikasi
    """)
    st.stop()

try:
    # Load model components untuk Speaker Identification
    model = joblib.load(os.path.join(models_dir, "speaker_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "speaker_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "speaker_label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "speaker_feature_names.pkl"))
    metadata = joblib.load(os.path.join(models_dir, "speaker_metadata.pkl"))
    
    # Classes dari label encoder
    classes = label_encoder.classes_  # ['nadia_buka', 'nadia_tutup', 'ufi_buka', 'ufi_tutup']
    
    # Authorized speakers
    AUTHORIZED_SPEAKERS = metadata.get('speakers', ['nadia', 'ufi'])
    CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence untuk accept
    
    st.success(f"✅ Model berhasil dimuat dari: `{models_dir}`")
    
except Exception as e:
    st.error(f"❌ Error loading model files: {type(e).__name__}: {str(e)}")
    st.info(f"📁 Direktori ditemukan: {models_dir}")
    st.warning("""
    **File yang dibutuhkan:**
    - speaker_model.pkl
    - speaker_scaler.pkl
    - speaker_label_encoder.pkl
    - speaker_feature_names.pkl
    - speaker_metadata.pkl
    
    **Cek apakah semua file ada:**
    ```bash
    ls -la tugas/models2/
    ```
    """)
    
    # Debug info
    with st.expander("🔍 Debug Information"):
        st.code(f"Models directory: {models_dir}")
        if os.path.exists(models_dir):
            st.write("Files in directory:")
            try:
                files = os.listdir(models_dir)
                for f in files:
                    st.write(f"  - {f}")
            except:
                st.write("  (Cannot list files)")
        st.code(f"Error: {str(e)}")
    
    st.stop()

# ===============================
# Fungsi Ekstraksi Fitur Audio - SPEAKER IDENTIFICATION
# ===============================
def extract_comprehensive_features(audio, sr):
    """
    Ekstraksi features untuk speaker identification & action classification
    ~100 features (sesuai dengan training model)
    """
    features = {}
    
    # ===== 1. TIME DOMAIN FEATURES =====
    features['mean'] = np.mean(audio)
    features['std'] = np.std(audio)
    features['max'] = np.max(audio)
    features['min'] = np.min(audio)
    features['median'] = np.median(audio)
    features['variance'] = np.var(audio)
    features['skewness'] = stats.skew(audio)
    features['kurtosis'] = stats.kurtosis(audio)
    features['range'] = np.ptp(audio)
    features['iqr'] = stats.iqr(audio)
    
    # Energy features
    features['energy'] = np.sum(audio**2)
    features['rms'] = np.sqrt(np.mean(audio**2))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Temporal features
    features['duration'] = len(audio) / sr
    gradient = np.gradient(audio)
    features['mean_gradient'] = np.mean(np.abs(gradient))
    features['max_gradient'] = np.max(np.abs(gradient))
    
    # Percentiles
    features['percentile_25'] = np.percentile(audio, 25)
    features['percentile_75'] = np.percentile(audio, 75)
    
    # Autocorrelation
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    features['autocorr_max'] = np.max(autocorr[1:100]) / autocorr[0] if autocorr[0] != 0 else 0
    
    # ===== 2. SPECTRAL FEATURES =====
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_centroid_max'] = np.max(spectral_centroids)
    features['spectral_centroid_min'] = np.min(spectral_centroids)
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # ===== 3. MFCC (CRITICAL for Speaker ID) =====
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
    
    # ===== 4. PITCH/F0 (Speaker Characteristic) =====
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_max'] = np.max(f0_clean)
            features['pitch_min'] = np.min(f0_clean)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0
            features['pitch_range'] = 0
    except:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_max'] = 0
        features['pitch_min'] = 0
        features['pitch_range'] = 0
    
    # ===== 5. CHROMA FEATURES =====
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    return features

def extract_audio_features(audio_data, sr=22050):
    """
    Wrapper function untuk ekstraksi features dengan format yang benar
    """
    try:
        # Ekstrak all features
        features_dict = extract_comprehensive_features(audio_data, sr)
        
        # Convert to DataFrame dengan urutan yang benar
        features_df = pd.DataFrame([features_dict])
        
        # Pastikan semua features ada (missing features diisi 0)
        for fname in feature_names:
            if fname not in features_df.columns:
                features_df[fname] = 0
        
        # Reorder columns sesuai feature_names
        features_df = features_df[feature_names]
        
        # Handle missing/inf values
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df.values[0]
    
    except Exception as e:
        st.error(f"Error saat ekstraksi fitur: {str(e)}")
        return None

# ===============================
# Header Aplikasi
# ===============================
st.title("🎤 Sistem Identifikasi Suara dengan Speaker Verification")

st.markdown(f"""
<div style='background-color: #020203; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #1f77b4; margin-top: 0;'>📌 Tentang Aplikasi</h3>
    <p>Sistem ini dapat mengidentifikasi <b>speaker</b> dan <b>perintah suara</b> secara bersamaan:</p>
    <ul>
        <li>👤 <b>Speaker Identification:</b> Mengenali suara dari <b>{', '.join([s.capitalize() for s in AUTHORIZED_SPEAKERS])}</b></li>
        <li>🎬 <b>Action Recognition:</b> Mendeteksi perintah <b>"Buka"</b> atau <b>"Tutup"</b></li>
        <li>🔒 <b>Security:</b> Menolak suara dari orang yang tidak terdaftar</li>
    </ul>
    <p><b>🎯 Akurasi Model:</b> {metadata.get('test_accuracy', 0)*100:.1f}% | 
       <b>📊 Total Classes:</b> {len(classes)} | 
       <b>👥 Authorized Speakers:</b> {len(AUTHORIZED_SPEAKERS)}</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Tab untuk Input
# ===============================
tab1, tab2, tab3 = st.tabs(["🎵 Upload Audio", "🎙️ Rekam Suara", "ℹ️ Informasi Model"])

with tab1:
    st.subheader("Upload File Audio")
    st.markdown("📁 **Format yang didukung:** WAV, MP3, OGG, FLAC")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Upload file audio yang berisi suara 'Buka' atau 'Tutup' dari speaker yang terdaftar"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"✅ File '{uploaded_file.name}' berhasil diupload! Klik tombol di bawah untuk analisis.")
        
        # Info sidebar
        with st.sidebar:
            st.info(f"""
            **📋 File Info:**
            - Nama: {uploaded_file.name}
            - Ukuran: {uploaded_file.size / 1024:.1f} KB
            
            **👥 Authorized Speakers:**
            {chr(10).join([f'- {s.capitalize()}' for s in AUTHORIZED_SPEAKERS])}
            
            **💡 Tips:**
            - Audio jelas tanpa noise
            - Durasi 1-5 detik
            - Confidence ≥ {CONFIDENCE_THRESHOLD}%
            """)
        
        # Tombol analisis di level utama (bukan dalam kolom)
        if st.button("🔍 Analisis Audio & Verifikasi Speaker", type="primary", use_container_width=True, key="analyze_upload"):
            with st.spinner("Menganalisis audio & memverifikasi speaker..."):
                try:
                    # Load audio
                    audio_bytes = uploaded_file.read()
                    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=5)
                    
                    # Ekstraksi fitur
                    features = extract_audio_features(audio_data, sr)
                    
                    if features is not None:
                        # Normalisasi features
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        
                        # Prediksi
                        prediction = model.predict(features_scaled)[0]
                        prediction_label = label_encoder.inverse_transform([prediction])[0]
                        
                        # Get confidence
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_scaled)[0]
                            confidence = np.max(proba) * 100
                        else:
                            confidence = None
                        
                        # Parse label (format: speaker_action)
                        if '_' in prediction_label:
                            speaker, action = prediction_label.split('_')
                        else:
                            speaker, action = 'unknown', 'unknown'
                        
                        # Speaker Verification
                        is_authorized = speaker in AUTHORIZED_SPEAKERS
                        is_confident = confidence is None or confidence >= CONFIDENCE_THRESHOLD
                        
                        # Tampilkan hasil
                        st.success("✅ Analisis Selesai!")
                        
                        # Result card dengan speaker verification
                        if is_authorized and is_confident:
                            # AUTHORIZED
                            if action.lower() == "buka":
                                result_color = "#28a745"
                                icon = "🔓"
                                status_icon = "✅"
                                status_text = "AUTHORIZED"
                            else:
                                result_color = "#dc3545"
                                icon = "🔒"
                                status_icon = "✅"
                                status_text = "AUTHORIZED"
                            
                            st.markdown(f"""
                            <div style='background-color: {result_color}; padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
                                <h1 style='margin: 0; font-size: 4em;'>{icon}</h1>
                                <h2 style='margin: 10px 0;'>Perintah Terdeteksi:</h2>
                                <h1 style='margin: 0; font-size: 3em;'>{action.upper()}</h1>
                                <h3 style='margin: 15px 0;'>{status_icon} Speaker: {speaker.upper()}</h3>
                                <p style='margin: 0; font-size: 1.1em; opacity: 0.9;'>{status_text}</p>
                                {f"<p style='margin-top: 10px; font-size: 1.2em;'>Confidence: {confidence:.1f}%</p>" if confidence else ""}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        else:
                            # UNAUTHORIZED
                            st.markdown(f"""
                            <div style='background-color: #ff6b6b; padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
                                <h1 style='margin: 0; font-size: 4em;'>🚫</h1>
                                <h2 style='margin: 10px 0;'>AKSES DITOLAK</h2>
                                <h3 style='margin: 15px 0;'>❌ UNAUTHORIZED SPEAKER</h3>
                                <p style='margin: 10px 0; font-size: 1.1em;'>
                                    Detected: {speaker.upper() if speaker != 'unknown' else 'UNKNOWN'}<br>
                                    {f"Confidence: {confidence:.1f}%" if confidence else ""}
                                </p>
                                <p style='margin-top: 15px; font-size: 0.95em; opacity: 0.9;'>
                                    ⚠️ Hanya speaker terdaftar yang dapat menggunakan sistem ini<br>
                                    Authorized speakers: {', '.join([s.capitalize() for s in AUTHORIZED_SPEAKERS])}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detail prediksi
                        with st.expander("📊 Detail Prediksi"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("👤 Speaker", speaker.upper())
                                with col_b:
                                    st.metric("🎬 Action", action.upper())
                                with col_c:
                                    if confidence:
                                        st.metric("💯 Confidence", f"{confidence:.1f}%")
                                
                                st.markdown("---")
                                st.markdown(f"**Predicted Label:** `{prediction_label}`")
                                st.markdown(f"**Authorized:** {'✅ Yes' if is_authorized else '❌ No'}")
                                st.markdown(f"**Confidence Check:** {'✅ Pass' if is_confident else f'❌ Below threshold ({CONFIDENCE_THRESHOLD}%)'}")
                                
                                if confidence and hasattr(model, 'predict_proba'):
                                    st.markdown("---")
                                    st.markdown("**Probability Distribution:**")
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': proba * 100
                                    }).sort_values('Probability', ascending=False)
                                    st.dataframe(prob_df, use_container_width=True)
                        
                        # Detail fitur
                        with st.expander("🔬 Lihat Detail Fitur Audio"):
                            st.markdown(f"**Total Features:** {len(features)}")
                            df_features = pd.DataFrame({
                                'Fitur': feature_names[:len(features)],
                                'Nilai': features
                            })
                            st.dataframe(df_features, use_container_width=True, height=400)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

with tab2:
    st.subheader("🎙️ Rekam Suara Manual")
    st.markdown("🎤 **Rekam suara Anda langsung dari browser**")
    
    # Audio recorder menggunakan st_audiorec
    try:
        from st_audiorec import st_audiorec
        
        st.info("""
        **📋 Cara Menggunakan:**
        1. Klik tombol **"Click to record"** untuk mulai merekam
        2. Ucapkan kata **"Buka"** atau **"Tutup"**
        3. Klik lagi untuk berhenti
        4. Klik tombol **"Analisis"** untuk memproses
        """)
        
        # Record audio
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            # Display audio player
            st.audio(wav_audio_data, format='audio/wav')
            st.success("✅ Rekaman berhasil! Klik tombol di bawah untuk analisis.")
            
            # Info di sidebar
            with st.sidebar:
                st.markdown("### 🎙️ Info Rekaman")
                st.info(f"""
                **👥 Authorized Speakers:**
                {chr(10).join([f'- {s.capitalize()}' for s in AUTHORIZED_SPEAKERS])}
                
                **💡 Tips:**
                - Rekam di tempat tenang
                - Ucapan jelas & keras
                - Durasi 1-3 detik
                - Confidence ≥ {CONFIDENCE_THRESHOLD}%
                """)
            
            st.markdown("---")
            
            # TOMBOL ANALISIS - BESAR & JELAS
            analyze_button = st.button(
                "🔍 ANALISIS REKAMAN & VERIFIKASI SPEAKER", 
                type="primary", 
                use_container_width=True,
                key="btn_analyze_recording"
            )
            
            if analyze_button:
                with st.spinner("🔄 Menganalisis rekaman & memverifikasi speaker..."):
                    try:
                        # Load audio from bytes
                        audio_data, sr = librosa.load(io.BytesIO(wav_audio_data), sr=22050, duration=5)
                        
                        # Ekstraksi fitur
                        features = extract_audio_features(audio_data, sr)
                        
                        if features is not None:
                            # Normalisasi features
                            features_scaled = scaler.transform(features.reshape(1, -1))
                            
                            # Prediksi
                            prediction = model.predict(features_scaled)[0]
                            prediction_label = label_encoder.inverse_transform([prediction])[0]
                            
                            # Get confidence
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(features_scaled)[0]
                                confidence = np.max(proba) * 100
                            else:
                                confidence = None
                            
                            # Parse label (format: speaker_action)
                            if '_' in prediction_label:
                                speaker, action = prediction_label.split('_')
                            else:
                                speaker, action = 'unknown', 'unknown'
                            
                            # Speaker Verification
                            is_authorized = speaker in AUTHORIZED_SPEAKERS
                            is_confident = confidence is None or confidence >= CONFIDENCE_THRESHOLD
                            
                            # Tampilkan hasil
                            st.success("✅ Analisis Selesai!")
                            
                            # Result card dengan speaker verification
                            if is_authorized and is_confident:
                                # AUTHORIZED
                                if action.lower() == "buka":
                                    result_color = "#28a745"
                                    icon = "🔓"
                                    status_icon = "✅"
                                    status_text = "AUTHORIZED"
                                else:
                                    result_color = "#dc3545"
                                    icon = "🔒"
                                    status_icon = "✅"
                                    status_text = "AUTHORIZED"
                                
                                st.markdown(f"""
                                <div style='background-color: {result_color}; padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
                                    <h1 style='margin: 0; font-size: 4em;'>{icon}</h1>
                                    <h2 style='margin: 10px 0;'>Perintah Terdeteksi:</h2>
                                    <h1 style='margin: 0; font-size: 3em;'>{action.upper()}</h1>
                                    <h3 style='margin: 15px 0;'>{status_icon} Speaker: {speaker.upper()}</h3>
                                    <p style='margin: 0; font-size: 1.1em; opacity: 0.9;'>{status_text}</p>
                                    {f"<p style='margin-top: 10px; font-size: 1.2em;'>Confidence: {confidence:.1f}%</p>" if confidence else ""}
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                # UNAUTHORIZED
                                st.markdown(f"""
                                <div style='background-color: #ff6b6b; padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
                                    <h1 style='margin: 0; font-size: 4em;'>🚫</h1>
                                    <h2 style='margin: 10px 0;'>AKSES DITOLAK</h2>
                                    <h3 style='margin: 15px 0;'>❌ UNAUTHORIZED SPEAKER</h3>
                                    <p style='margin: 10px 0; font-size: 1.1em;'>
                                        Detected: {speaker.upper() if speaker != 'unknown' else 'UNKNOWN'}<br>
                                        {f"Confidence: {confidence:.1f}%" if confidence else ""}
                                    </p>
                                    <p style='margin-top: 15px; font-size: 0.95em; opacity: 0.9;'>
                                        ⚠️ Hanya speaker terdaftar yang dapat menggunakan sistem ini<br>
                                        Authorized speakers: {', '.join([s.capitalize() for s in AUTHORIZED_SPEAKERS])}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Detail prediksi
                            with st.expander("📊 Detail Prediksi"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("👤 Speaker", speaker.upper())
                                with col_b:
                                    st.metric("🎬 Action", action.upper())
                                with col_c:
                                    if confidence:
                                        st.metric("💯 Confidence", f"{confidence:.1f}%")
                                
                                st.markdown("---")
                                st.markdown(f"**Predicted Label:** `{prediction_label}`")
                                st.markdown(f"**Authorized:** {'✅ Yes' if is_authorized else '❌ No'}")
                                st.markdown(f"**Confidence Check:** {'✅ Pass' if is_confident else f'❌ Below threshold ({CONFIDENCE_THRESHOLD}%)'}")
                                
                                if confidence and hasattr(model, 'predict_proba'):
                                    st.markdown("---")
                                    st.markdown("**Probability Distribution:**")
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': proba * 100
                                    }).sort_values('Probability', ascending=False)
                                    st.dataframe(prob_df, use_container_width=True)
                            
                            # Detail fitur
                            with st.expander("🔬 Lihat Detail Fitur Audio"):
                                st.markdown(f"**Total Features:** {len(features)}")
                                df_features = pd.DataFrame({
                                    'Fitur': feature_names[:len(features)],
                                    'Nilai': features
                                })
                                st.dataframe(df_features, use_container_width=True, height=400)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            # Tampilkan info di sidebar saat belum merekam
            with st.sidebar:
                st.markdown("### 🎙️ Info Rekaman")
                st.info(f"""
                **👥 Authorized Speakers:**
                {chr(10).join([f'- {s.capitalize()}' for s in AUTHORIZED_SPEAKERS])}
                
                **💡 Tips:**
                - Rekam di tempat tenang
                - Ucapan jelas & keras
                - Durasi 1-3 detik
                - Confidence ≥ {CONFIDENCE_THRESHOLD}%
                """)
    
    except ImportError:
        st.warning("⚠️ **Library 'st_audiorec' belum terinstall**")
        st.markdown("""
        Untuk menggunakan fitur rekam suara, install library berikut:
        
        ```bash
        pip install streamlit-audiorec
        ```
        
        Atau gunakan tab **Upload Audio** untuk upload file audio.
        """)

with tab3:
    st.subheader("ℹ️ Informasi Model - Speaker Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Detail Model")
        st.info(f"""
        **Purpose:** Speaker Identification & Action Classification
        
        **Model Type:** {metadata.get('model_type', type(model).__name__)}
        
        **Test Accuracy:** {metadata.get('test_accuracy', 0) * 100:.1f}%
        
        **CV Score:** {metadata.get('cv_score', 0) * 100:.1f}%
        
        **Total Features:** {metadata.get('n_features', len(feature_names))}
        
        **Total Classes:** {len(classes)} classes
        
        **Authorized Speakers:** {', '.join([s.capitalize() for s in AUTHORIZED_SPEAKERS])}
        
        **Actions:** {', '.join(metadata.get('actions', ['buka', 'tutup']))}
        
        **Training Samples:** {metadata.get('n_samples_train', 'N/A')}
        
        **Testing Samples:** {metadata.get('n_samples_test', 'N/A')}
        
        **Confidence Threshold:** {CONFIDENCE_THRESHOLD}%
        """)
        
        st.markdown("### 🎯 Classes")
        for i, cls in enumerate(classes, 1):
            speaker, action = cls.split('_')
            st.markdown(f"{i}. **{cls}** → 👤 {speaker.upper()} | 🎬 {action.upper()}")
    
    with col2:
        st.markdown("### 🎵 Features Ekstraksi")
        st.markdown("""
        **Time Domain (19 features):**
        - Statistical: mean, std, variance, skewness, kurtosis
        - Energy: energy, RMS, Zero Crossing Rate
        - Temporal: duration, gradient, percentiles
        
        **Spectral Features (~80 features):**
        - Spectral Centroid, Rolloff, Bandwidth, Contrast
        - **MFCC (20 coefficients x 3 stats)** ⭐ Critical for Speaker ID
        - **Pitch/F0 (5 features)** ⭐ Voice characteristics
        - Chroma features (pitch class)
        
        **Total: ~100 features**
        """)
        
        st.markdown("### 🔒 Security Features")
        st.markdown(f"""
        - ✅ **Speaker Verification**: Only authorized speakers accepted
        - ✅ **Confidence Threshold**: Minimum {CONFIDENCE_THRESHOLD}% confidence
        - ✅ **Multi-factor**: Speaker + Action verification
        - ✅ **Rejection**: Unauthorized speakers are rejected
        """)
    
    st.markdown("---")
    st.markdown("### 🔬 Cara Kerja Sistem")
    st.markdown("""
    1. **Input Audio** → Upload atau rekam audio
    2. **Feature Extraction** → Ekstraksi ~100 features (MFCC, Pitch, Spectral, dll)
    3. **Normalization** → Features dinormalisasi dengan StandardScaler
    4. **Classification** → Model memprediksi speaker + action
    5. **Verification** → Cek authorization:
       - ✅ Speaker harus terdaftar (Nadia/Ufi)
       - ✅ Confidence ≥ 70%
    6. **Output** → Accept/Reject + display speaker & action
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    if hasattr(model, 'feature_importances_'):
        st.markdown("**Feature Importance (Top 10):**")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index('Feature'))

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>🎤 Sistem Identifikasi Suara dengan Speaker Verification</b></p>
    <p>👥 Authorized Speakers: {', '.join([s.capitalize() for s in AUTHORIZED_SPEAKERS])} | 
       🎬 Actions: Buka/Tutup | 
       🔒 Security: Speaker + Confidence Verification</p>
    <p>Dibuat menggunakan Streamlit & Machine Learning | © 2025</p>
</div>
""", unsafe_allow_html=True)
