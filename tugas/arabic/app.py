import streamlit as st
import numpy as np
import librosa
import json
import pickle
from tensorflow import keras
import io
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Arabic Digit Recognition",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model dan metadata
@st.cache_resource
def load_model_and_metadata():
    """Memuat model, scaler, dan metadata"""
    model = None
    scaler = None
    metadata = None
    
    try:
        # Memuat metadata
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Memuat model
        model = keras.models.load_model('best_model.h5')
        
        # Memuat scaler jika ada
        if os.path.exists('scaler.pkl'):
            try:
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as scaler_error:
                st.warning(f"⚠️ Scaler file is corrupted or invalid. Continuing without scaler normalization. Error: {str(scaler_error)}")
                scaler = None
        
        return model, scaler, metadata
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Fungsi untuk ekstraksi fitur MFCC
def extract_mfcc(audio_data, sr, n_mfcc=13, max_length=93):
    """Ekstraksi fitur MFCC dari audio"""
    try:
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Padding atau truncate untuk mendapatkan panjang yang konsisten
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        return mfcc.T  # Transpose untuk mendapatkan shape (max_length, n_mfcc)
    except Exception as e:
        st.error(f"Error extracting MFCC: {str(e)}")
        return None

# Fungsi untuk prediksi
def predict_digit(model, audio_features, scaler=None):
    """Melakukan prediksi digit dari fitur audio"""
    try:
        # Normalisasi SEBELUM reshape jika scaler tersedia
        if scaler is not None:
            # Normalisasi pada level 2D (timesteps, features)
            normalized_features = scaler.transform(audio_features)
        else:
            normalized_features = audio_features
        
        # Reshape untuk input model (batch_size, timesteps, features)
        features = normalized_features.reshape(1, normalized_features.shape[0], normalized_features.shape[1])
        
        # Prediksi
        predictions = model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Fungsi untuk membuat visualisasi waveform
def plot_waveform(audio_data, sr):
    """Membuat plot waveform audio"""
    time = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', name='Waveform'))
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=300
    )
    return fig

# Fungsi untuk membuat visualisasi MFCC
def plot_mfcc(mfcc):
    """Membuat plot heatmap MFCC"""
    fig = px.imshow(
        mfcc.T,
        aspect='auto',
        color_continuous_scale='Viridis',
        labels=dict(x="Time Frame", y="MFCC Coefficient", color="Value")
    )
    fig.update_layout(title='MFCC Features', height=400)
    return fig

# Fungsi untuk membuat visualisasi probabilitas
def plot_probabilities(probabilities, class_names):
    """Membuat bar chart untuk probabilitas prediksi"""
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=probabilities, marker_color='lightblue')
    ])
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Digit Class',
        yaxis_title='Probability',
        height=400
    )
    return fig

# Header aplikasi
st.markdown('<div class="main-header">Arabic Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Pengenalan Angka Arab menggunakan Deep Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microphone.png", width=100)
    st.title("Menu Navigasi")
    
    page = st.radio(
        "Pilih Halaman:",
        ["Home", "Prediksi Audio", "Informasi Model", "Tentang"]
    )
    
    st.markdown("---")
    st.markdown("### Pengaturan")
    show_visualizations = st.checkbox("Tampilkan Visualisasi", value=True)
    show_confidence = st.checkbox("Tampilkan Confidence Score", value=True)
    
    st.markdown("---")
    st.markdown("### Informasi")
    st.info(f"Tanggal: {datetime.now().strftime('%d %B %Y')}")

# Memuat model
model, scaler, metadata = load_model_and_metadata()

# Halaman Home
if page == "Home":
    st.markdown("## Selamat Datang!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Akurasi Model</h3>
            <h2 style="color: #28a745;">99.4%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Jumlah Kelas</h3>
            <h2 style="color: #1f77b4;">10 Digit</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Type</h3>
            <h2 style="color: #ff7f0e;">1D CNN</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>Cara Menggunakan Aplikasi</h3>
        <ol>
            <li>Pilih halaman <strong>"Prediksi Audio"</strong> di sidebar</li>
            <li>Upload file audio format WAV</li>
            <li>Klik tombol <strong>"Prediksi"</strong></li>
            <li>Lihat hasil prediksi dan visualisasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Fitur Utama")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Prediksi Real-time**
        - Upload dan prediksi langsung
        - Hasil instan dengan confidence score
        
        **Visualisasi Lengkap**
        - Waveform audio
        - MFCC features
        - Probability distribution
        """)
    
    with col2:
        st.markdown("""
        **Model Akurat**
        - Akurasi 99.4%
        - F1-Score 99.4%
        
        **User-Friendly**
        - Interface intuitif
        - Navigasi mudah
        """)

# Halaman Prediksi Audio
elif page == "Prediksi Audio":
    st.markdown("## Prediksi Audio")
    
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang benar.")
    else:
        st.markdown('<div class="info-box">Pilih metode input: Upload file audio atau Rekam langsung dari mikrofon.</div>', unsafe_allow_html=True)
        
        # Tabs untuk Upload atau Record
        input_tab1, input_tab2 = st.tabs(["Upload File", "Rekam Audio"])
        
        audio_source = None
        
        with input_tab1:
            st.markdown("**Upload file audio (WAV, MP3, M4A, FLAC, OGG)**")
            uploaded_file = st.file_uploader(
                "Pilih file audio",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                help="Upload file audio dalam format WAV, MP3, M4A, FLAC, atau OGG",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file)
                audio_source = ("file", uploaded_file)
        
        with input_tab2:
            st.markdown("**Rekam digit Arab (0-9) langsung dari mikrofon**")
            st.info("Klik tombol rekam, ucapkan satu digit Arab (0-9), lalu klik stop.")
            
            # Initialize session state for recording
            if 'recording_key' not in st.session_state:
                st.session_state.recording_key = 0
            
            recorded_audio = st.audio_input("Rekam audio", key=f"audio_recorder_{st.session_state.recording_key}")
            
            if recorded_audio is not None:
                col_audio, col_reset = st.columns([3, 1])
                with col_audio:
                    st.audio(recorded_audio)
                with col_reset:
                    if st.button("Reset", type="secondary", use_container_width=True):
                        st.session_state.recording_key += 1
                        st.rerun()
                audio_source = ("recorded", recorded_audio)
        
        # Tombol Prediksi
        if audio_source is not None:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                predict_button = st.button("Prediksi", use_container_width=True, type="primary")
            
            if predict_button:
                with st.spinner('Memproses audio...'):
                    # Membaca audio dengan sampling rate yang sama dengan training data
                    source_type, audio_file = audio_source
                    audio_data, sr = librosa.load(io.BytesIO(audio_file.read()), sr=11025)
                    
                    # Reset file pointer if needed for replay
                    audio_file.seek(0)
                    
                    # Ekstraksi MFCC
                    mfcc_features = extract_mfcc(
                        audio_data, 
                        sr, 
                        n_mfcc=metadata['num_mfcc'], 
                        max_length=metadata['max_length']
                    )
                    
                    if mfcc_features is not None:
                        # Prediksi
                        predicted_class, confidence, probabilities = predict_digit(
                            model, mfcc_features, scaler
                        )
                        
                        if predicted_class is not None:
                            st.markdown("---")
                            
                            # Hasil Prediksi
                            st.markdown("### Hasil Prediksi")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="Digit Terdeteksi",
                                    value=f"{predicted_class}",
                                    delta=None
                                )
                            
                            if show_confidence:
                                with col2:
                                    st.metric(
                                        label="Confidence Score",
                                        value=f"{confidence * 100:.2f}%",
                                        delta=None
                                    )
                                
                                with col3:
                                    status = "Tinggi" if confidence > 0.9 else "Sedang" if confidence > 0.7 else "Rendah"
                                    st.metric(
                                        label="Status",
                                        value=status,
                                        delta=None
                                    )
                            
                            # Visualisasi
                            if show_visualizations:
                                st.markdown("---")
                                st.markdown("### Visualisasi")
                                
                                tab1, tab2, tab3 = st.tabs(["Waveform", "MFCC Features", "Probabilities"])
                                
                                with tab1:
                                    fig_waveform = plot_waveform(audio_data, sr)
                                    st.plotly_chart(fig_waveform, use_container_width=True)
                                
                                with tab2:
                                    fig_mfcc = plot_mfcc(mfcc_features)
                                    st.plotly_chart(fig_mfcc, use_container_width=True)
                                
                                with tab3:
                                    fig_prob = plot_probabilities(probabilities, metadata['class_names'])
                                    st.plotly_chart(fig_prob, use_container_width=True)
                                
                                # Tabel probabilitas
                                st.markdown("#### Probabilitas per Kelas")
                                prob_df = pd.DataFrame({
                                    'Digit': metadata['class_names'],
                                    'Probabilitas': [f"{p*100:.2f}%" for p in probabilities],
                                    'Nilai': probabilities
                                }).sort_values('Nilai', ascending=False)
                                st.dataframe(prob_df[['Digit', 'Probabilitas']], hide_index=True, use_container_width=True)

# Halaman Informasi Model
elif page == "Informasi Model":
    st.markdown("## Informasi Model")
    
    if metadata is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Spesifikasi Model")
            st.markdown(f"""
            - **Nama Model**: {metadata['model_name']}
            - **Jumlah Kelas**: {metadata['num_classes']}
            - **MFCC Coefficients**: {metadata['num_mfcc']}
            - **Max Sequence Length**: {metadata['max_length']}
            - **Training Date**: {metadata['training_date']}
            """)
        
        with col2:
            st.markdown("### Performa Model")
            st.markdown(f"""
            - **Test Accuracy**: {metadata['test_accuracy'] * 100:.2f}%
            - **Test F1-Score**: {metadata['test_f1_score'] * 100:.2f}%
            """)
            
            # Progress bars
            st.progress(metadata['test_accuracy'])
            st.caption("Accuracy")
            st.progress(metadata['test_f1_score'])
            st.caption("F1-Score")
        
        st.markdown("---")
        
        st.markdown("### Kelas yang Dikenali")
        
        # Display classes in a grid
        cols = st.columns(5)
        for idx, class_name in enumerate(metadata['class_names']):
            with cols[idx % 5]:
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 0.5rem 0;">
                    <h2>{class_name}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Arsitektur Model")
        st.markdown("""
        Model menggunakan **1D Convolutional Neural Network (CNN)** yang dioptimalkan untuk:
        - Ekstraksi fitur temporal dari audio
        - Klasifikasi multi-kelas dengan 10 output
        - Preprocessing dengan MFCC (Mel-frequency cepstral coefficients)
        """)
        
        # Model summary (jika tersedia)
        if model is not None:
            with st.expander("Lihat Model Summary"):
                # Create a string buffer to capture model.summary()
                from io import StringIO
                import sys
                
                buffer = StringIO()
                sys.stdout = buffer
                model.summary()
                sys.stdout = sys.__stdout__
                
                st.text(buffer.getvalue())

# Halaman Tentang
elif page == "Tentang":
    st.markdown("## Tentang Aplikasi")
    
    st.markdown("""
    <div class="info-box">
        <h3>Arabic Digit Recognition System</h3>
        <p>Sistem pengenalan angka Arab menggunakan teknologi Deep Learning yang mampu mengidentifikasi 
        digit 0-9 dari input audio dengan akurasi tinggi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Teknologi yang Digunakan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Framework & Libraries:**
        - Streamlit - Web Framework
        - TensorFlow/Keras - Deep Learning
        - Librosa - Audio Processing
        - Plotly - Visualisasi Interaktif
        - Pandas - Data Manipulation
        """)
    
    with col2:
        st.markdown("""
        **Fitur Utama:**
        - Real-time Prediction
        - Interactive Visualizations
        - High Accuracy (99.4%)
        - User-Friendly Interface
        - Comprehensive Analytics
        """)
    
    st.markdown("---")
    
    st.markdown("### Cara Kerja Sistem")
    
    st.markdown("""
    1. **Input Audio**
       - User upload file audio format WAV
       
    2. **Preprocessing**
       - Ekstraksi MFCC features
       - Normalisasi data
       - Padding/Truncating untuk konsistensi
       
    3. **Prediksi**
       - Model CNN memproses fitur
       - Menghasilkan probabilitas untuk setiap kelas
       - Menentukan digit dengan confidence score
       
    4. **Visualisasi**
       - Menampilkan waveform
       - Heatmap MFCC
       - Distribusi probabilitas
    """)
    
    st.markdown("---")
    
    st.markdown("### Developer Information")
    st.info("Aplikasi ini dibuat menggunakan Streamlit untuk mendemonstrasikan kemampuan Deep Learning dalam pengenalan suara.")
    
    st.markdown("### Version History")
    st.markdown("""
    - **v1.0.0** (Dec 2025)
        - Initial release
        - 1D CNN model
        - Interactive UI
        - Comprehensive visualizations
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Arabic Digit Recognition System | Powered by Deep Learning</p>
    <p>© 2025 - Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
