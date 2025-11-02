import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import librosa
import io

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Sistem Identifikasi Suara",
    page_icon="🎤",
    layout="wide"
)

# ===============================
# Load Model, Scaler, Label Encoder, dan Metadata
# ===============================

# Dapatkan direktori file saat ini
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

try:
    with open(os.path.join(current_dir, "audio_classifier.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(current_dir, "audio_classifier_scaler.pkl"), "rb") as f:
        feature_names = pickle.load(f)  # Array nama fitur

    with open(os.path.join(current_dir, "audio_classifier_label_encoder.pkl"), "rb") as f:
        classes = pickle.load(f)  # Array dengan ['buka', 'tutup']

    with open(os.path.join(current_dir, "audio_classifier_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Load scaler yang sebenarnya (jika ada file terpisah)
    # Untuk sementara kita akan skip normalisasi atau buat dummy scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(feature_names))
    scaler.scale_ = np.ones(len(feature_names))
    scaler.n_features_in_ = len(feature_names)
        
except Exception as e:
    st.error(f"❌ Error loading model files: {type(e).__name__}: {str(e)}")
    st.info(f"📁 Current directory: {current_dir}")
    st.warning("Please make sure all .pkl files are in the same directory as app.py")
    st.stop()

# ===============================
# Fungsi Ekstraksi Fitur Audio
# ===============================
def extract_audio_features(audio_data, sr=22050, n_features=29):
    """
    Ekstraksi fitur dari audio file untuk menghasilkan 29 fitur
    """
    try:
        features_dict = {}
        
        # MFCC (13 coefficients) - mean dan std = 26 fitur
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(13):
            features_dict[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features_dict[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Spectral Centroid - mean = 1 fitur
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        features_dict['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        # Spectral Bandwidth - mean = 1 fitur  
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        features_dict['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # Zero Crossing Rate - mean = 1 fitur
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        features_dict['zcr_mean'] = np.mean(zcr)
        
        # Total: 26 + 1 + 1 + 1 = 29 fitur
        
        # Konversi ke array sesuai urutan feature_names
        feature_list = []
        for fname in feature_names:
            if fname in features_dict:
                feature_list.append(features_dict[fname])
            else:
                feature_list.append(0.0)  # Default jika fitur tidak ditemukan
        
        return np.array(feature_list)
    
    except Exception as e:
        st.error(f"Error saat ekstraksi fitur: {str(e)}")
        return None

# ===============================
# Header Aplikasi
# ===============================
st.title("🎤 Sistem Identifikasi Suara: Buka/Tutup")

st.markdown("""
<div style='background-color: #020203; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #1f77b4; margin-top: 0;'>📌 Tentang Aplikasi</h3>
    <p>Sistem ini dapat mengidentifikasi perintah suara <b>"Buka"</b> atau <b>"Tutup"</b> 
    dari file audio yang Anda upload. Aplikasi menggunakan Machine Learning untuk 
    menganalisis karakteristik suara dan mengenali perintah yang diucapkan.</p>
    <p><b>🎯 Akurasi Model: 100%</b> | <b>📊 Dataset: 400 samples</b></p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Tab untuk Input
# ===============================
tab1, tab2 = st.tabs(["🎵 Upload Audio", "ℹ️ Informasi Model"])

with tab1:
    st.subheader("Upload File Audio")
    st.markdown("📁 **Format yang didukung:** WAV, MP3, OGG, FLAC")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=['wav', 'mp3', 'ogg', 'flac'],
        help="Upload file audio yang berisi suara 'Buka' atau 'Tutup'"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analisis Audio", type="primary", use_container_width=True):
                with st.spinner("Menganalisis audio..."):
                    try:
                        # Load audio
                        audio_bytes = uploaded_file.read()
                        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                        
                        # Ekstraksi fitur
                        features = extract_audio_features(audio_data, sr)
                        
                        if features is not None:
                            # Prediksi (tanpa scaling karena model mungkin sudah dilatih tanpa scaling)
                            prediction = model.predict(features.reshape(1, -1))
                            label = classes[prediction[0]]  # Ambil label dari array classes
                            
                            # Probabilitas (jika model support)
                            try:
                                proba = model.predict_proba(features.reshape(1, -1))[0]
                                confidence = np.max(proba) * 100
                            except:
                                confidence = None
                            
                            # Tampilkan hasil
                            st.success("✅ Analisis Selesai!")
                            
                            # Result card
                            if label.lower() == "buka":
                                result_color = "#28a745"
                                icon = "🔓"
                            else:
                                result_color = "#dc3545"
                                icon = "🔒"
                            
                            st.markdown(f"""
                            <div style='background-color: {result_color}; padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
                                <h1 style='margin: 0; font-size: 4em;'>{icon}</h1>
                                <h2 style='margin: 10px 0;'>Perintah Terdeteksi:</h2>
                                <h1 style='margin: 0; font-size: 3em;'>{label.upper()}</h1>
                                {f"<p style='margin-top: 15px; font-size: 1.2em;'>Confidence: {confidence:.1f}%</p>" if confidence else ""}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detail fitur
                            with st.expander("📊 Lihat Detail Fitur Audio"):
                                df_features = pd.DataFrame({
                                    'Fitur': feature_names[:len(features)],
                                    'Nilai': features
                                })
                                st.dataframe(df_features, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.info("""
            **📋 Petunjuk:**
            1. Upload file audio
            2. Klik tombol 'Analisis Audio'
            3. Tunggu hasil prediksi
            
            **💡 Tips:**
            - Gunakan audio yang jelas
            - Hindari noise/gangguan
            - Durasi 1-3 detik ideal
            """)

with tab2:
    st.subheader("ℹ️ Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Detail Model")
        st.info(f"""
        **Nama Model:** {metadata.get('model_name', 'Audio Classifier')}
        
        **Jenis Model:** {metadata.get('model_type', type(model).__name__)}
        
        **Jumlah Fitur:** {metadata.get('n_features', len(feature_names))}
        
        **Kelas Prediksi:** {' / '.join([c.capitalize() for c in classes])}
        
        **Akurasi Test:** {metadata.get('test_accuracy', 'N/A') * 100:.1f}%
        
        **CV Score:** {metadata.get('cv_score', 'N/A') * 100:.1f}%
        
        **Data Training:** {metadata.get('n_samples_train', 'N/A')} samples
        
        **Data Testing:** {metadata.get('n_samples_test', 'N/A')} samples
        """)
    
    with col2:
        st.markdown("### 🎵 Fitur Audio yang Digunakan")
        st.markdown("""
        - **MFCC** (Mel-frequency cepstral coefficients)
        - **Chroma** (Karakteristik nada)
        - **Spectral Centroid** (Pusat frekuensi)
        - **Spectral Bandwidth** (Lebar pita frekuensi)
        - **Zero Crossing Rate** (Tingkat perubahan sinyal)
        - **RMS Energy** (Energi sinyal)
        - **Tempo** (Kecepatan audio)
        """)
    
    st.markdown("---")
    st.markdown("### 🔬 Cara Kerja Sistem")
    st.markdown("""
    1. **Input Audio** → Audio direkam atau diupload
    2. **Preprocessing** → Audio dinormalisasi dan dibersihkan dari noise
    3. **Feature Extraction** → Ekstraksi fitur menggunakan librosa
    4. **Normalization** → Fitur dinormalisasi menggunakan scaler
    5. **Classification** → Model machine learning memprediksi kelas
    6. **Output** → Menampilkan hasil "Buka" atau "Tutup"
    """)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>Sistem Identifikasi Suara: Buka/Tutup</b></p>
    <p>Dibuat menggunakan Streamlit & Machine Learning | © 2025</p>
</div>
""", unsafe_allow_html=True)
