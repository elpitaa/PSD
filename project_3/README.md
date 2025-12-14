# Arabic Digit Classification - Project 3

## Deskripsi
Sistem klasifikasi otomatis untuk mengenali digit berbahasa Arab (0-9) dari rekaman suara menggunakan MFCC features dan Deep Learning.

## Struktur Folder
```
project_3/
├── app.py                   # Streamlit web application
├── best_model.h5           # Model terbaik (CNN/LSTM)
├── scaler.pkl              # StandardScaler untuk normalisasi
├── model_metadata.json     # Metadata model dan metrik
└── README.md               # File ini
```

## Dataset
- **Sumber**: Spoken Arabic Digits (UCI ML Repository)
- **Total Samples**: 8,800 (6,600 train, 2,200 test)
- **Features**: 13 MFCC coefficients per frame
- **Sampling Rate**: 11,025 Hz
- **Classes**: 10 digits (0-9)

## Model
- **Type**: 1D CNN / LSTM (pilih yang terbaik)
- **Input Shape**: (93, 13) - 93 timesteps × 13 MFCC
- **Output**: 10 classes (softmax)
- **Performance**: Akurasi ~99%, F1-Score ~99%

## Cara Menggunakan

### 1. Training Model (via Jupyter Notebook)
```bash
cd /workspaces/PSD/tugas
jupyter notebook arabic_digit_classification.ipynb
```
Run semua cells untuk training model. Model akan otomatis tersimpan di `project_3/`.

### 2. Running Streamlit App
```bash
cd /workspaces/PSD/project_3
streamlit run app.py
```

App akan running di http://localhost:8501

### 3. Upload Audio untuk Prediksi
- Format: WAV, MP3, M4A, FLAC, OGG
- Rekam atau upload file audio yang berisi ucapan digit Arab (0-9)
- App akan otomatis:
  - Resample audio ke 11,025 Hz
  - Ekstrak MFCC features
  - Normalisasi dengan scaler
  - Prediksi digit

## Critical Notes

⚠️ **Sampling Rate Harus Konsisten!**
- Dataset dilatih dengan sr=11,025 Hz
- Saat inference, HARUS resample ke sr=11,025 Hz
- App.py sudah dikonfigurasi dengan benar: `librosa.load(audio_file, sr=11025)`

## Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
streamlit
librosa
plotly
soundfile
```

## Files Required untuk Deployment
1. ✅ `best_model.h5` atau `best_model.pkl` - Model
2. ✅ `scaler.pkl` - Normalization scaler
3. ✅ `model_metadata.json` - Metadata
4. ✅ `app.py` - Streamlit application

## Author
Data Science Student

## Metodologi
CRISP-DM (Cross-Industry Standard Process for Data Mining)

## Last Updated
December 14, 2025
