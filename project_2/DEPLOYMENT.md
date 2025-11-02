# 🚀 Panduan Deploy ke Streamlit Cloud

## Langkah-langkah Deploy

### 1. Persiapan Repository GitHub

1. **Buat repository baru** di GitHub
2. **Upload semua file** project_2:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Sistem Identifikasi Suara Buka/Tutup"
   git remote add origin <URL_REPOSITORY>
   git push -u origin main
   ```

### 2. File yang Harus Ada di Repository

✅ Pastikan file-file ini ada:
- `app.py` - Aplikasi utama
- `requirements.txt` - Dependencies
- `runtime.txt` - Python version
- `audio_classifier.pkl` - Model
- `audio_classifier_scaler.pkl` - Feature names
- `audio_classifier_label_encoder.pkl` - Classes
- `audio_classifier_metadata.pkl` - Metadata
- `README.md` - Dokumentasi

### 3. Deploy di Streamlit Cloud

1. Buka https://share.streamlit.io/
2. Login dengan akun GitHub
3. Klik **"New app"**
4. Pilih repository Anda
5. Pilih branch: `main`
6. Main file path: `app.py`
7. Klik **"Deploy!"**

### 4. Konfigurasi (Opsional)

Jika diperlukan, buat file `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

## 🔧 Troubleshooting

### Error: Module not found
- Pastikan semua library ada di `requirements.txt`
- Cek versi library yang kompatibel

### Error: File not found
- Pastikan semua file .pkl ada di repository
- Cek path file di kode

### App terlalu lambat
- Optimize ekstraksi fitur audio
- Gunakan caching dengan `@st.cache_data`

## 📊 Monitoring

Setelah deploy, Anda bisa:
- Melihat logs di dashboard Streamlit Cloud
- Monitor usage dan performance
- Update app dengan push ke GitHub

## 🔄 Update Aplikasi

Untuk update aplikasi:
```bash
git add .
git commit -m "Update: deskripsi perubahan"
git push
```

Streamlit Cloud akan otomatis rebuild aplikasi.

## 🌐 URL Aplikasi

Setelah deploy, aplikasi akan tersedia di:
```
https://<username>-<repo-name>-<branch>.streamlit.app
```

## 💡 Tips Deployment

1. **Ukuran File**: Pastikan file .pkl tidak terlalu besar (< 100MB)
2. **Dependencies**: Gunakan versi library yang stabil
3. **Testing**: Test di local sebelum deploy
4. **Documentation**: Buat README yang jelas
5. **Git LFS**: Untuk file besar (> 50MB), gunakan Git LFS

## 🎯 Checklist Deploy

- [ ] Semua file .pkl ada dan valid
- [ ] requirements.txt lengkap
- [ ] runtime.txt sesuai
- [ ] App berjalan di local
- [ ] README.md informatif
- [ ] .gitignore configured
- [ ] Repository public di GitHub
- [ ] Deploy di Streamlit Cloud
- [ ] Test aplikasi yang sudah deploy
- [ ] Share link ke user

## 📝 Catatan

- Deploy pertama mungkin memakan waktu 3-5 menit
- Streamlit Cloud gratis dengan batasan resources
- Untuk production, pertimbangkan Streamlit Cloud Pro

---

**Status Deploy:** ✅ Ready to Deploy!
