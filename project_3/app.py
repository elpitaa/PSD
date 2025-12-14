"""
Arabic Digit Recognition - Streamlit Application
Aplikasi pengenalan digit berbahasa Arab (0-9) dari rekaman suara
Menggunakan Deep Learning (1D CNN/LSTM) dengan MFCC features
Version: 1.0.1
"""

# Standard library imports
import os
import io
import json
import pickle
import warnings
from datetime import datetime

# Third-party imports
import streamlit as st
import numpy as np
import pandas as pd
import librosa
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ========================================
# GLOBAL CONFIGURATION
# ========================================
# Get base directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Arabic Digit Recognition",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/elpitaa/PSD',
        'Report a bug': 'https://github.com/elpitaa/PSD/issues',
        'About': "# Arabic Digit Recognition\nPengenalan digit berbahasa Arab menggunakan Deep Learning"
    }
)

# ========================================
# LOAD MODEL DAN RESOURCES
# ========================================
@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Memuat model, scaler, dan metadata dengan error handling"""
    with st.spinner('Loading model and resources...'):
        model = None
        scaler = None
        metadata = None
        
        try:
            # Use global BASE_DIR
            
            # Memuat metadata
            metadata_path = os.path.join(BASE_DIR, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                st.error(f"File model_metadata.json tidak ditemukan di {BASE_DIR}!")
                return None, None, None
            
            # Memuat model (prioritas SVM .pkl, fallback ke .h5)
            model_pkl_path = os.path.join(BASE_DIR, 'best_model.pkl')
            model_h5_path = os.path.join(BASE_DIR, 'best_model.h5')
            
            if os.path.exists(model_pkl_path):
                with open(model_pkl_path, 'rb') as f:
                    model = pickle.load(f)
            elif os.path.exists(model_h5_path):
                model = keras.models.load_model(model_h5_path)
            else:
                st.error(f"File model tidak ditemukan di {BASE_DIR}!")
                return None, None, None
            
            # Memuat scaler
            scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                st.warning(f"File scaler.pkl tidak ditemukan di {BASE_DIR}. Prediksi mungkin tidak akurat!")
            
            return model, scaler, metadata
            
        except Exception as e:
            st.error(f"Error loading resources: {str(e)}")
            return None, None, None

# ========================================
# AUDIO PROCESSING FUNCTIONS
# ========================================
def extract_mfcc(audio_data, sr, n_mfcc=13, max_length=93):
    """Ekstraksi fitur MFCC dari audio"""
    try:
        # Ekstraksi MFCC dengan sampling rate 11,025 Hz
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Padding atau truncate
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        return mfcc.T  # Transpose: (max_length, n_mfcc)
    except Exception as e:
        st.error(f"Error extracting MFCC: {str(e)}")
        return None

def predict_digit(model, audio_features, scaler=None):
    """Melakukan prediksi digit dari fitur audio"""
    try:
        # Cek apakah model adalah SVM (sklearn) atau Keras model
        if hasattr(model, 'predict_proba'):  # SVM/sklearn model
            # Flatten features untuk SVM (harus 1D per sample)
            features_flat = audio_features.flatten().reshape(1, -1)
            
            # Normalisasi dengan scaler
            if scaler is not None:
                normalized_features = scaler.transform(features_flat)
            else:
                normalized_features = features_flat
            
            predictions = model.predict_proba(normalized_features)[0]
            predicted_class = model.predict(normalized_features)[0]
            confidence = predictions[predicted_class]
        else:  # Keras/Deep Learning model
            # Normalisasi dengan scaler
            if scaler is not None:
                normalized_features = scaler.transform(audio_features)
            else:
                normalized_features = audio_features
            
            # Reshape untuk input model (batch_size, timesteps, features)
            features = normalized_features.reshape(1, normalized_features.shape[0], normalized_features.shape[1])
            predictions = model.predict(features, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence, predictions
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# ========================================
# VISUALIZATION FUNCTIONS
# ========================================
def plot_waveform(audio_data, sr):
    """Membuat plot waveform audio dengan Plotly"""
    time = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, 
        y=audio_data, 
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1)
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=300,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_mfcc(mfcc):
    """Membuat heatmap MFCC features"""
    fig = px.imshow(
        mfcc.T,
        aspect='auto',
        color_continuous_scale='Viridis',
        labels=dict(x="Time Frame", y="MFCC Coefficient", color="Value")
    )
    
    fig.update_layout(
        title='🎵 MFCC Features Heatmap',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_probabilities(probabilities, class_names):
    """Membuat bar chart probabilitas prediksi"""
    colors = ['#667eea' if p == max(probabilities) else '#e0e0e0' for p in probabilities]
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_names, 
            y=probabilities, 
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Digit Class',
        yaxis_title='Probability',
        height=400,
        template='plotly_white',
        showlegend=False,
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig

def create_digit_table():
    """Membuat tabel digit Arab"""
    data = {
        'Digit': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'Arab': ['صفر', 'واحد', 'اثنان', 'ثلاثة', 'أربعة', 'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة'],
        'Pengucapan': ['Sifr', 'Wahid', 'Itnan', 'Thalatha', 'Arba\'a', 'Khamsa', 'Sitta', 'Sab\'a', 'Thamaniya', 'Tis\'a']
    }
    return pd.DataFrame(data)

# ========================================
# MAIN APPLICATION
# ========================================
def main():
    # Header
    st.markdown('<h1 class="main-header">Arabic Digit Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistem Pengenalan Digit Berbahasa Arab (0-9) menggunakan Deep Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, metadata = load_model_and_metadata()
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("Arabic Digit")
        st.caption("Speech Recognition")
        
        st.markdown("---")
        
        st.markdown("### Menu")
        page = st.radio(
            "Pilih Halaman:",
            ["Home", "Prediksi Audio", "Informasi Model", "Tentang Dataset", "Panduan"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("### Status Model")
        if metadata:
            st.markdown(f"""
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">Model Type</div>
                    <div class="sidebar-metric-value">{metadata.get('model_name', 'Unknown')}</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">Test Accuracy</div>
                    <div class="sidebar-metric-value">{metadata.get('test_accuracy', 0)*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="sidebar-metric">
                    <div class="sidebar-metric-label">F1-Score</div>
                    <div class="sidebar-metric-value">{metadata.get('test_f1_score', 0)*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model tidak tersedia")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### Pengaturan Tampilan")
        show_visualizations = st.checkbox("Visualisasi", value=True, help="Tampilkan grafik waveform dan MFCC")
        show_confidence = st.checkbox("Confidence Score", value=True, help="Tampilkan tingkat kepercayaan prediksi")
        show_probabilities = st.checkbox("Probabilitas Detail", value=True, help="Tampilkan probabilitas semua kelas")
        
        st.markdown("---")
        
        # Info
        st.markdown("### Info")
        st.caption("Dataset: Spoken Arabic Digits")
        st.caption("Samples: 8,800 utterances")
        st.caption("Speakers: 88 (balanced)")
        st.caption("MFCC: 13 coefficients")
        
        st.markdown("---")
        st.caption("© 2025 Arabic Digit Recognition")
        st.caption("Methodology: CRISP-DM")
    
    # ========================================
    # PAGE: HOME
    # ========================================
    if page == "Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">Akurasi Model</h3>
                    <h1 style="color: #007bff;">97.2%</h1>
                    <p>Test Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">Jumlah Kelas</h3>
                    <h1 style="color: #007bff;">10 Digit</h1>
                    <p>Arabic Digits (0-9)</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #333;">Model Type</h3>
                    <h1 style="color: #007bff;">SVM</h1>
                    <p>Machine Learning</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Start Guide
        st.markdown("### Quick Start Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4>Cara Menggunakan</h4>
                    <ol>
                        <li>Pilih menu <b>"Prediksi Audio"</b> di sidebar</li>
                        <li>Pilih digit (0-9) yang ingin ditest</li>
                        <li>Pilih nomor sample (1-220)</li>
                        <li>Klik tombol <b>"Prediksi"</b></li>
                        <li>Lihat hasil prediksi dan visualisasi</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="success-box">
                    <h4>Fitur Utama</h4>
                    <ul>
                        <li>Test dengan 2,200 samples dari dataset</li>
                        <li>220 samples per digit (0-9)</li>
                        <li>Visualisasi probability distribution</li>
                        <li>Akurasi 97.2% (Test set)</li>
                        <li>Hasil reproducible dan reliable</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Dataset Overview
        st.markdown("---")
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", "8,800", help="6,600 training + 2,200 testing")
        col2.metric("Speakers", "88", help="44 laki-laki + 44 perempuan")
        col3.metric("MFCC Features", "13", help="Mel-Frequency Cepstral Coefficients")
        col4.metric("Sampling Rate", "11,025 Hz", help="Telephone quality")
        
        # Arabic Digits Table
        st.markdown("---")
        st.markdown("### Digit Arab (0-9)")
        df_digits = create_digit_table()
        st.dataframe(df_digits, use_container_width=True, hide_index=True)
    
    # ========================================
    # PAGE: PREDIKSI AUDIO
    # ========================================
    elif page == "Prediksi Audio":
        st.markdown("## Prediksi Audio")
        
        if model is None:
            st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang benar.")
            st.stop()
        
        st.markdown("""
            <div class="info-box">
                <b>Petunjuk:</b> Pilih digit (0-9) dan nomor sample dari test dataset untuk melihat prediksi model.
                Dataset ini berisi 2,200 samples audio yang sudah diproses menjadi MFCC features.
            </div>
        """, unsafe_allow_html=True)
        
        # INFO BOX
        st.success("""
            **Model Performance**: Aplikasi ini menggunakan test dataset dengan 220 samples per digit (total 2,200 samples).
            Model mencapai akurasi **97.2%** pada dataset ini, memastikan prediksi yang reliable dan reproducible.
        """)
        st.markdown("### Test Model dengan Dataset Asli")
        st.info("Pilih sample dari test dataset untuk melihat prediksi yang akurat")
        
        # Load test data
        # @st.cache_data  # Temporarily disabled for deployment debugging
        def load_test_data(base_dir):
                import os
                """Load test dataset with error handling for production"""
                # Check if test data exists
                test_data_path = os.path.join(base_dir, 'data', 'Test_Arabic_Digit.txt')
                
                if not os.path.exists(test_data_path):
                    st.error(f"File test data tidak ditemukan di: {test_data_path}")
                    st.info("Silakan gunakan fitur 'Upload Audio' atau 'Rekam Audio'.")
                    return None, None
                
                def parse_file(filepath, blocks_per_digit):
                    sequences = []
                    current_sequence = []
                    
                    with open(filepath, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                if len(current_sequence) > 0:
                                    sequences.append(np.array(current_sequence))
                                    current_sequence = []
                                continue
                            try:
                                values = [float(x) for x in line.split()]
                                if len(values) == 13:
                                    current_sequence.append(values)
                            except:
                                pass
                    
                    if len(current_sequence) > 0:
                        sequences.append(np.array(current_sequence))
                    
                    labels = []
                    for digit in range(10):
                        labels.extend([digit] * blocks_per_digit)
                    
                    return sequences, np.array(labels[:len(sequences)])
                
                X_test_raw, y_test = parse_file(test_data_path, blocks_per_digit=220)
                
                # Pad sequences
                def pad_sequences(X, max_len):
                    X_padded = []
                    for x in X:
                        if x.shape[0] < max_len:
                            pad_width = max_len - x.shape[0]
                            x_padded = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')
                        else:
                            x_padded = x[:max_len, :]
                        X_padded.append(x_padded)
                    return np.array(X_padded)
                
                X_test = pad_sequences(X_test_raw, 93)
                return X_test, y_test
            
            X_test, y_test = load_test_data(BASE_DIR)
            
            # Skip this tab if test data not available
            if X_test is None:
            st.warning("Dataset tidak dapat dimuat. Silakan periksa file data.")
            
            # Calculate index
            test_idx = selected_digit * 220 + (sample_number - 1)
            
            st.markdown(f"""
                **Sample Info:**
                - Digit: **{selected_digit}**
                - Sample: **{sample_number}/220**
                - Index: {test_idx}
            """)
            
            if st.button("PREDIKSI DARI DATASET", use_container_width=True, type="primary"):
                with st.spinner("Memproses data..."):
                    sample = X_test[test_idx]
                    true_label = y_test[test_idx]
                    
                    # Normalize dan predict - support both SVM and CNN
                    if hasattr(model, 'predict_proba'):  # SVM
                        # Flatten untuk SVM
                        sample_flat = sample.flatten().reshape(1, -1)
                        normalized = scaler.transform(sample_flat)
                        predictions = model.predict_proba(normalized)[0]
                        predicted_class = model.predict(normalized)[0]
                        confidence = predictions[predicted_class]
                    else:  # CNN
                        normalized = scaler.transform(sample)
                        features = normalized.reshape(1, 93, 13)
                        predictions = model.predict(features, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### Hasil Prediksi")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("True Label", true_label)
                    with col2:
                        st.metric("Predicted", predicted_class)
                    with col3:
                        is_correct = "CORRECT" if predicted_class == true_label else "WRONG"
                        st.metric("Status", is_correct)
                    
                    st.success(f"**Confidence:** {confidence*100:.2f}%")
                    
                    # Probability distribution
                    st.markdown("### Distribusi Probabilitas")
                    prob_df = pd.DataFrame({
                        'Digit': [str(i) for i in range(10)],
                        'Probability': predictions * 100
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Digit', 
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='Viridis',
                        title='Probabilitas Setiap Digit'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # PAGE: INFORMASI MODEL
    # ========================================
    elif page == "Informasi Model":
        st.markdown("## Informasi Model")
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Details")
                st.json({
                    "Model Name": metadata.get('model_name', 'Unknown'),
                    "Model Type": "Machine Learning (SVM)",
                    "Input Shape": f"({metadata.get('max_length', 93)} × {metadata.get('num_mfcc', 13)})",
                    "Output Classes": metadata.get('num_classes', 10),
                    "Training Date": metadata.get('training_date', 'Unknown')
                })
            
            with col2:
                st.markdown("### Performance Metrics")
                
                accuracy = metadata.get('test_accuracy', 0) * 100
                f1_score = metadata.get('test_f1_score', 0) * 100
                
                st.metric("Test Accuracy", f"{accuracy:.2f}%")
                st.metric("F1-Score (Macro)", f"{f1_score:.2f}%")
                
                # Progress bars
                st.markdown("**Accuracy Progress**")
                st.progress(accuracy / 100)
                
                st.markdown("**F1-Score Progress**")
                st.progress(f1_score / 100)
            
            st.markdown("---")
            
            # Model Architecture
            st.markdown("### Model Architecture")
            
            st.code("""
SVM (Support Vector Machine) Architecture:
├── Input: Flattened MFCC features (93 × 13 = 1,209 features)
├── Kernel: RBF (Radial Basis Function)
├── C (Regularization): Optimized via GridSearchCV
├── Gamma: Optimized via GridSearchCV
└── Output: 10 classes (softmax probability)

Advantages:
✓ Excellent generalization
✓ Effective in high-dimensional space
✓ Memory efficient
✓ Robust to overfitting
            """, language="text")
            
            # Feature Extraction
            st.markdown("### Feature Extraction")
            st.info("""
            **MFCC (Mel-Frequency Cepstral Coefficients)**
            - 13 koefisien per frame
            - Sampling rate: 11,025 Hz
            - Max timesteps: 93 frames
            - Normalization: StandardScaler
            """)
        else:
            st.warning("Metadata tidak tersedia")
    
    # ========================================
    # PAGE: TENTANG DATASET
    # ========================================
    elif page == "Tentang Dataset":
        st.markdown("## Tentang Dataset")
        
        st.markdown("""
        ### Spoken Arabic Digits Dataset
        
        Dataset ini berisi rekaman suara digit berbahasa Arab (0-9) yang sudah diekstrak
        menjadi fitur MFCC (Mel-Frequency Cepstral Coefficients).
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", "8,800")
            st.metric("Training Set", "6,600")
            st.metric("Test Set", "2,200")
        
        with col2:
            st.metric("Speakers", "88")
            st.metric("Gender", "Balanced")
            st.metric("Repetitions", "10x per digit")
        
        with col3:
            st.metric("MFCC Features", "13")
            st.metric("Max Frames", "93")
            st.metric("Sampling Rate", "11,025 Hz")
        
        st.markdown("---")
        
        # Dataset characteristics
        tab1, tab2, tab3 = st.tabs(["Karakteristik", "Digit Arab", "Distribusi"])
        
        with tab1:
            st.markdown("""
            ### Kelebihan Dataset
            
            - **Balanced**: Semua digit memiliki jumlah sample yang sama (880 per digit)
            - **Gender Balanced**: 50% laki-laki, 50% perempuan
            - **Sufficient Size**: 8,800 samples cukup untuk training
            - **Pre-processed**: MFCC sudah diekstrak, siap pakai
            - **Clean Split**: Train-test split sudah tersedia
            - **Real Speakers**: Bukan synthetic, tapi rekaman orang asli
            
            ### Keterbatasan
            
            - **Limited Speakers**: Hanya 88 orang
            - **Single Environment**: Rekaman dalam kondisi controlled
            - **Telephone Quality**: Sampling rate 11,025 Hz (bukan high-quality)
            """)
        
        with tab2:
            df_digits = create_digit_table()
            st.dataframe(df_digits, use_container_width=True, hide_index=True)
            
            st.markdown("""
            ### Catatan Pengucapan
            - Setiap digit diucapkan 10 kali oleh setiap speaker
            - Total: 88 speakers × 10 digits × 10 repetitions = **8,800 utterances**
            - Variasi aksen dan kecepatan bicara
            """)
        
        with tab3:
            # Create sample distribution chart
            samples_per_digit = [880] * 10
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(10)),
                    y=samples_per_digit,
                    marker_color='#667eea',
                    text=samples_per_digit,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Distribusi Samples per Digit',
                xaxis_title='Digit',
                yaxis_title='Jumlah Samples',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Dataset perfectly balanced - semua digit memiliki jumlah sample yang sama!")
    
    # ========================================
    # PAGE: PANDUAN
    # ========================================
    elif page == "Panduan":
        st.markdown("## Panduan Penggunaan")
        
        # FAQ
        st.markdown("### Frequently Asked Questions (FAQ)")
        
        with st.expander("Format audio apa yang didukung?"):
            st.markdown("""
            Aplikasi mendukung format audio berikut:
            - WAV (Recommended)
            - MP3
            - M4A
            - FLAC
            - OGG
            
            **Rekomendasi**: Gunakan WAV untuk hasil terbaik.
            """)
        
        with st.expander("Berapa panjang audio yang ideal?"):
            st.markdown("""
            - **Minimal**: 0.5 detik
            - **Maksimal**: 3 detik
            - **Ideal**: 1-2 detik
            
            Audio terlalu pendek atau terlalu panjang mungkin mempengaruhi akurasi.
            """)
        
        with st.expander("Bagaimana cara merekam audio yang baik?"):
            st.markdown("""
            **Tips Merekam Audio:**
            
            1. Pastikan lingkungan tenang (minimal noise)
            2. Dekatkan mikrofon ke mulut (jarak 10-15 cm)
            3. Ucapkan digit dengan jelas dan tidak terlalu cepat
            4. Ucapkan hanya satu digit per rekaman
            5. Durasi 1-2 detik sudah cukup
            """)
        
        with st.expander("Mengapa prediksi tidak akurat?"):
            st.markdown("""
            **Kemungkinan penyebab:**
            
            1. Audio terlalu noisy/berisik
            2. Pengucapan kurang jelas
            3. Kualitas audio rendah
            4. Background noise terlalu besar
            5. Format audio tidak didukung dengan baik
            
            **Solusi:**
            - Rekam ulang dengan kondisi lebih baik
            - Gunakan format WAV
            - Pastikan pengucapan jelas
            """)
        
        with st.expander("Apa itu Confidence Score?"):
            st.markdown("""
            **Confidence Score** adalah tingkat kepercayaan model terhadap prediksinya.
            
            - **90-100%**: Sangat Tinggi (model sangat yakin)
            - **70-90%**: Tinggi (model cukup yakin)
            - **<70%**: Rendah (model kurang yakin)
            
            Jika confidence score rendah, coba rekam ulang dengan kondisi lebih baik.
            """)
        
        st.markdown("---")
        
        # Technical Details
        st.markdown("### Technical Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Information:**
            - Type: SVM (Machine Learning)
            - Input: MFCC features (93 × 13 = 1,209 features)
            - Output: 10 classes (digit 0-9)
            - Framework: Scikit-learn
            """)
        
        with col2:
            st.markdown("""
            **Feature Extraction:**
            - MFCC: 13 coefficients
            - Sampling Rate: 11,025 Hz
            - Window: Hamming
            - Normalization: StandardScaler
            """)
        
        st.markdown("---")
        
        # Contact & Support
        st.markdown("### Contact & Support")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Email**
            
            support@example.com
            """)
        
        with col2:
            st.markdown("""
            **GitHub**
            
            [elpitaa/PSD](https://github.com/elpitaa/PSD)
            """)
        
        with col3:
            st.markdown("""
            **Documentation**
            
            [Read the Docs](#)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Arabic Digit Recognition | Built with Streamlit</p>
            <p>© 2025 | Methodology: CRISP-DM | Model: 1D CNN</p>
        </div>
    """, unsafe_allow_html=True)

# ========================================
# RUN APPLICATION
# ========================================
if __name__ == "__main__":
    main()
