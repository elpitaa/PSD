# 🎤 Arabic Digit Recognition - Streamlit App

Aplikasi web interaktif untuk pengenalan angka Arab (0-9) menggunakan Deep Learning dan Streamlit.

## ✨ Fitur Utama

- 🎯 **Prediksi Real-time**: Upload audio dan dapatkan hasil prediksi instan
- 📊 **Visualisasi Lengkap**: Waveform, MFCC features, dan probability distribution
- 🎨 **UI Modern**: Interface yang clean, intuitif, dan responsive
- 📈 **Akurasi Tinggi**: Model dengan akurasi 99.4%
- 🔧 **Mudah Dikembangkan**: Kode terstruktur dan well-documented

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Akses Aplikasi

Buka browser dan akses: `http://localhost:8501`

## 📁 Struktur Proyek

```
arabic/
│
├── app.py                    # Main Streamlit application
├── best_model.h5            # Trained 1D CNN model
├── model_metadata.json      # Model configuration and metrics
├── scaler.pkl               # Feature scaler (optional)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎯 Cara Menggunakan

1. **Pilih Halaman** di sidebar:
   - 🏠 **Home**: Overview dan panduan
   - 🎵 **Prediksi Audio**: Upload dan prediksi file audio
   - 📊 **Informasi Model**: Detail model dan performa
   - ℹ️ **Tentang**: Informasi aplikasi

2. **Upload Audio**:
   - Klik halaman "Prediksi Audio"
   - Upload file WAV
   - Klik tombol "Prediksi"

3. **Lihat Hasil**:
   - Digit yang terdeteksi
   - Confidence score
   - Visualisasi lengkap

## 🛠️ Teknologi

- **Streamlit**: Web framework untuk Python
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

## 📊 Performa Model

- **Model Type**: 1D Convolutional Neural Network
- **Test Accuracy**: 99.41%
- **F1-Score**: 99.41%
- **Classes**: 10 (digits 0-9)

## 🔧 Konfigurasi

Aplikasi menggunakan konfigurasi dari `model_metadata.json`:

```json
{
  "model_name": "1D CNN",
  "num_classes": 10,
  "num_mfcc": 13,
  "max_length": 93,
  "test_accuracy": 0.994,
  "test_f1_score": 0.994
}
```

## 📝 Pengembangan

### Menambah Fitur Baru

1. Tambahkan halaman baru di sidebar navigation
2. Buat fungsi handler untuk halaman tersebut
3. Gunakan komponen Streamlit yang tersedia

### Memodifikasi Model

1. Update file `best_model.h5`
2. Sesuaikan `model_metadata.json`
3. Restart aplikasi

## 🎨 Komponen Streamlit yang Digunakan

- **Layout**: `st.sidebar`, `st.columns`, `st.tabs`
- **Input**: `st.file_uploader`, `st.button`, `st.checkbox`, `st.radio`
- **Output**: `st.metric`, `st.audio`, `st.plotly_chart`, `st.dataframe`
- **Styling**: `st.markdown`, Custom CSS
- **Caching**: `@st.cache_resource` untuk optimasi

## 📄 Lisensi

Project ini dibuat untuk tujuan edukatif dan demonstrasi.

## 👨‍💻 Kontribusi

Silakan fork repository ini dan submit pull request untuk improvement.

---

**Happy Coding! 🚀**
