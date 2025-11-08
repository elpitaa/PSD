# 🚀 Quick Start Guide - Speaker Verification App

## Langkah Cepat

### 1️⃣ Train Model (Jika Belum)

```bash
cd /workspaces/PSD/tugas
# Buka dan jalankan notebook: Identifikasi_Suara_Buka_Tutup.ipynb
# Jalankan semua cells hingga model tersimpan di folder models/
```

### 2️⃣ Copy Model Files

```bash
# Dari root directory
cd /workspaces/PSD

# Buat folder models di project_2 jika belum ada
mkdir -p project_2/models

# Copy semua file model
cp tugas/models/speaker_model*.pkl project_2/models/
```

### 3️⃣ Jalankan Aplikasi

```bash
cd project_2
streamlit run app.py
```

### 4️⃣ Test Aplikasi

1. Buka browser: `http://localhost:8501`
2. Upload file audio test dari:
   - `tugas/audio_data2/BukaTutup_nadia/buka1.mp3` (harus AUTHORIZED)
   - `tugas/audio_data2/BukaTutup_ufi/Rekaman/Buka/*.wav` (harus AUTHORIZED)
3. Lihat hasil prediksi & verification

---

## ✅ Checklist

- [ ] Model sudah ditraining (run notebook)
- [ ] File `.pkl` ada di `project_2/models/`
- [ ] Dependencies sudah terinstall (`pip install -r requirements.txt`)
- [ ] Aplikasi berjalan tanpa error
- [ ] Test dengan audio Nadia → Result: AUTHORIZED
- [ ] Test dengan audio Ufi → Result: AUTHORIZED
- [ ] Test dengan audio lain → Result: UNAUTHORIZED (jika bukan Nadia/Ufi)

---

## 🎯 Expected Results

### Test dengan Audio Nadia (buka1.mp3)
```
✅ AUTHORIZED
👤 Speaker: NADIA
🎬 Action: BUKA
💯 Confidence: >70%
```

### Test dengan Audio Ufi
```
✅ AUTHORIZED
👤 Speaker: UFI
🎬 Action: BUKA/TUTUP
💯 Confidence: >70%
```

### Test dengan Audio Orang Lain
```
❌ UNAUTHORIZED
⚠️ Speaker tidak terdaftar
```

---

## 📁 File Structure

```
project_2/
├── models/                    ← Files dari training
│   ├── speaker_model.pkl
│   ├── speaker_model_scaler.pkl
│   ├── speaker_model_label_encoder.pkl
│   ├── speaker_model_feature_names.pkl
│   └── speaker_model_metadata.pkl
├── app.py                     ← Aplikasi Streamlit (SUDAH DIUPDATE)
├── requirements.txt
└── README_SPEAKER_VERIFICATION.md
```

---

## 🔧 Troubleshooting

### Error: Model files not found
```bash
# Cek apakah folder models ada
ls project_2/models/

# Jika kosong, copy dari tugas/models/
cp tugas/models/speaker_model*.pkl project_2/models/
```

### Error: Module not found
```bash
# Install dependencies
pip install -r requirements.txt
```

### Model tidak load
```bash
# Cek apakah semua 5 file ada
ls -la project_2/models/speaker_model*.pkl
# Harus ada 5 file
```

---

## 💡 Tips

1. **Training model dulu**: Jangan lupa jalankan notebook untuk generate file `.pkl`
2. **Copy files**: Pastikan semua 5 file `.pkl` ter-copy ke folder `models/`
3. **Test sistematis**: Test dengan sample audio yang pasti (Nadia & Ufi)
4. **Cek confidence**: Jika <70%, coba audio yang lebih jelas

---

**Ready to Go!** 🚀
