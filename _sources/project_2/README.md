# 🎤 Sistem Identifikasi Suara: Buka/Tutup

Aplikasi web untuk mengidentifikasi perintah suara "Buka" atau "Tutup" menggunakan Machine Learning.

## 📋 Deskripsi

Sistem ini menggunakan Random Forest Classifier untuk mengklasifikasikan audio menjadi dua kategori:
- **Buka** 🔓
- **Tutup** 🔒

Model dilatih dengan 29 fitur audio yang diekstraksi menggunakan librosa, termasuk MFCC, spectral features, dan lainnya.

## 🎯 Performa Model

- **Akurasi Test:** 100%
- **CV Score:** 100%
- **Model:** RandomForestClassifier
- **Jumlah Fitur:** 29
- **Dataset:**
  - Training: 320 samples
  - Testing: 80 samples

## 🚀 Cara Menjalankan Aplikasi

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

## 📦 Requirements

- streamlit==1.38.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.4.2
- joblib==1.3.2
- librosa==0.10.1
- soundfile==0.12.1

## 🎵 Fitur Audio yang Diekstraksi

Model menggunakan 29 fitur audio:

1. **MFCC (Mel-frequency cepstral coefficients)**: 13 koefisien (mean & std) = 26 fitur
   - Menangkap karakteristik spektral audio
   
2. **Spectral Centroid (mean)**: 1 fitur
   - Mengukur "pusat massa" dari spektrum
   
3. **Spectral Bandwidth (mean)**: 1 fitur
   - Mengukur lebar pita frekuensi
   
4. **Zero Crossing Rate (mean)**: 1 fitur
   - Mengukur tingkat perubahan tanda dalam sinyal

## 📱 Fitur Aplikasi

### 1. Upload Audio
- Upload file audio (WAV, MP3, OGG, FLAC)
- Preview audio sebelum analisis
- Ekstraksi fitur otomatis
- Prediksi dengan confidence score
- Visualisasi hasil yang menarik

### 2. Input Manual Fitur
- Mode advanced untuk eksperimen
- Input 29 fitur secara manual
- Berguna untuk testing dan debugging

### 3. Informasi Model
- Detail lengkap tentang model
- Penjelasan fitur audio
- Cara kerja sistem

## 📂 Struktur File

```
project_2/
├── app.py                              # Aplikasi Streamlit
├── audio_classifier.pkl                # Model ML (Random Forest)
├── audio_classifier_scaler.pkl         # Nama-nama fitur (29 fitur)
├── audio_classifier_label_encoder.pkl  # Label classes ['buka', 'tutup']
├── audio_classifier_metadata.pkl       # Metadata model
├── requirements.txt                    # Dependencies
├── runtime.txt                         # Python version
└── README.md                          # Dokumentasi
```

## 🔬 Cara Kerja

1. **Input Audio** → User upload file audio
2. **Preprocessing** → Audio dinormalisasi
3. **Feature Extraction** → Ekstraksi 29 fitur menggunakan librosa
4. **Classification** → Model Random Forest memprediksi kelas
5. **Output** → Menampilkan hasil "Buka" atau "Tutup" dengan confidence

## 💡 Tips Penggunaan

- Gunakan audio yang jelas dan berkualitas baik
- Hindari background noise
- Durasi audio ideal: 1-3 detik
- Format WAV memberikan hasil terbaik

## 🛠️ Teknologi

- **Streamlit** - Web framework
- **Librosa** - Audio feature extraction
- **Scikit-learn** - Machine learning
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation

## 📝 Catatan

Model ini dilatih dengan dataset khusus untuk bahasa Indonesia, kata "Buka" dan "Tutup". 
Untuk hasil terbaik, gunakan audio dengan karakteristik serupa dengan data training.

## 👨‍💻 Developer

Dibuat dengan ❤️ menggunakan Streamlit & Machine Learning

© 2025 - Proyek Sain Data

---

**Status:** ✅ Ready for Deployment
