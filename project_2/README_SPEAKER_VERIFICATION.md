# 🎤 Sistem Identifikasi Suara dengan Speaker Verification

Aplikasi Streamlit untuk identifikasi suara dengan fitur **Speaker Verification** yang dapat mengenali speaker tertentu dan mengklasifikasikan perintah suara.

## ✨ Fitur Utama - UPDATE TERBARU

### 🎯 Dual Classification
- **Speaker Identification**: Mengenali suara dari speaker yang terdaftar (Nadia & Ufi)
- **Action Recognition**: Mengklasifikasi perintah "Buka" atau "Tutup"

### 🔒 Security Features
- ✅ **Speaker Verification**: Hanya menerima input dari speaker yang sudah terdaftar
- ✅ **Confidence Threshold**: Minimum 70% confidence untuk accept
- ✅ **Rejection System**: Otomatis menolak speaker yang tidak dikenal
- ✅ **Multi-factor Verification**: Speaker + Action + Confidence

### 📊 Model Capabilities
- **Total Classes**: 4 kombinasi (nadia_buka, nadia_tutup, ufi_buka, ufi_tutup)
- **Features**: ~100 features (MFCC, Pitch, Spectral, Time Domain)
- **Accuracy**: >90% (tergantung hasil training)
- **Real-time**: Prediksi dalam hitungan detik

## 🚀 Cara Menjalankan

### 1. Persiapan File Model

Pastikan folder `models/` berisi file-file berikut (hasil training dari notebook):

```
project_2/
├── models/
│   ├── speaker_model.pkl
│   ├── speaker_model_scaler.pkl
│   ├── speaker_model_label_encoder.pkl
│   ├── speaker_model_feature_names.pkl
│   └── speaker_model_metadata.pkl
├── app.py
└── requirements.txt
```

**PENTING**: File-file `.pkl` didapat dari menjalankan notebook `Identifikasi_Suara_Buka_Tutup.ipynb` di folder `tugas/`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Copy Model Files

Copy semua file dari `tugas/models/` ke `project_2/models/`:

```bash
# Dari root directory PSD
cp tugas/models/speaker_model*.pkl project_2/models/
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📱 Cara Menggunakan Aplikasi

### Upload & Analisis
1. **Upload Audio**: Pilih file audio (.wav, .mp3, .ogg, .flac)
2. **Klik Analisis**: Tekan tombol "Analisis Audio & Verifikasi Speaker"
3. **Lihat Hasil**: Sistem akan menampilkan:
   - ✅ **AUTHORIZED** - Jika speaker terdaftar & confidence tinggi
   - ❌ **UNAUTHORIZED** - Jika speaker tidak dikenal atau confidence rendah

### Interpretasi Hasil

#### ✅ Authorized (Hijau/Merah)
- **Speaker**: Nadia atau Ufi
- **Action**: Buka (🔓 hijau) atau Tutup (🔒 merah)
- **Status**: AUTHORIZED
- **Confidence**: ≥70%

#### ❌ Unauthorized (Merah)
- **Speaker**: Unknown atau tidak terdaftar
- **Status**: AKSES DITOLAK
- **Alasan**: Speaker tidak terdaftar atau confidence <70%

## 🎵 Features yang Diekstrak

### 1. Time Domain Features (19)
- Statistik: mean, std, variance, skewness, kurtosis
- Energy: total energy, RMS, Zero Crossing Rate
- Temporal: duration, gradient, percentiles, autocorrelation

### 2. Spectral Features (~80)
- **MFCC (20 coefficients × 3 stats)**: Critical untuk speaker identification
- **Pitch/F0 (5 features)**: Karakteristik unik suara speaker
- Spectral: centroid, rolloff, bandwidth, contrast
- Chroma: pitch class features

**Total: ~100 features** untuk akurasi maksimal

## 🔧 Konfigurasi

### Authorized Speakers
Edit di `app.py`:
```python
AUTHORIZED_SPEAKERS = metadata.get('speakers', ['nadia', 'ufi'])
```

### Confidence Threshold
Edit di `app.py`:
```python
CONFIDENCE_THRESHOLD = 70.0  # Minimum confidence (0-100)
```

## 📊 Model Information

### Training
- **Dataset**: Audio files dari 2 speakers (Nadia & Ufi)
- **Samples**: ~400+ audio files total
- **Classes**: 4 classes (speaker_action combinations)
- **Features**: ~100 features per audio

### Model Files
1. **speaker_model.pkl**: Trained classifier (Random Forest/SVM/etc)
2. **speaker_model_scaler.pkl**: StandardScaler untuk normalisasi
3. **speaker_model_label_encoder.pkl**: Label encoder untuk decoding
4. **speaker_model_feature_names.pkl**: Urutan features
5. **speaker_model_metadata.pkl**: Info model (accuracy, params, dll)

## 🔍 Troubleshooting

### Error: Model files not found
**Solusi**: Pastikan file `.pkl` ada di folder `models/` atau di direktori yang sama dengan `app.py`

### Error saat ekstraksi fitur
**Solusi**: 
- Pastikan audio tidak corrupt
- Cek format audio (gunakan .wav untuk hasil terbaik)
- Durasi audio 1-5 detik ideal

### Confidence rendah
**Solusi**:
- Gunakan audio dengan kualitas baik
- Hindari background noise
- Pastikan speaker berbicara jelas
- Coba rekam ulang dengan kondisi lebih tenang

### Speaker tidak dikenali
**Solusi**:
- Pastikan speaker adalah Nadia atau Ufi
- Jika speaker baru, perlu re-training model dengan data speaker tersebut
- Cek confidence score (harus ≥70%)

## 🎓 Technical Details

### Classification Pipeline
```
Audio Input
    ↓
Feature Extraction (~100 features)
    ↓
Normalization (StandardScaler)
    ↓
Model Prediction
    ↓
Speaker Verification
    ↓
Accept / Reject
```

### Security Layer
```python
is_authorized = speaker in ['nadia', 'ufi']
is_confident = confidence >= 70.0

if is_authorized and is_confident:
    # ACCEPT
else:
    # REJECT
```

## 📝 Perubahan dari Versi Sebelumnya

### Versi Lama (2 classes)
- Classes: buka, tutup
- Features: 29 features
- Tidak ada speaker verification

### Versi Baru (4 classes) ⭐
- Classes: nadia_buka, nadia_tutup, ufi_buka, ufi_tutup
- Features: ~100 features
- Speaker verification: ✅
- Confidence threshold: ✅
- Rejection system: ✅

## 🚀 Deployment

### Local
```bash
streamlit run app.py
```

### Cloud (Streamlit Cloud)
1. Push ke GitHub
2. Connect ke Streamlit Cloud
3. Deploy dengan `app.py` sebagai main file
4. Pastikan folder `models/` ter-upload

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📄 License

MIT License - Feel free to use and modify

## 👥 Contributors

- Model Training: Tugas/Identifikasi_Suara_Buka_Tutup.ipynb
- App Development: project_2/app.py
- Dataset: Nadia & Ufi audio recordings

---

**Status**: ✅ Production Ready with Speaker Verification  
**Last Updated**: November 2025
