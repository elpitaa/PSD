# 🎤 Cara Menggunakan Aplikasi

## 📱 Untuk User/Pengguna

### Metode 1: Upload File Audio (Recommended)

1. **Buka aplikasi** di browser
2. **Pilih tab "Upload Audio"**
3. **Klik tombol "Browse files"**
4. **Pilih file audio** Anda (WAV, MP3, OGG, atau FLAC)
5. **Preview audio** untuk memastikan file benar
6. **Klik "Analisis Audio"**
7. **Tunggu beberapa detik**
8. **Lihat hasil prediksi**: 
   - 🔓 **BUKA** (background hijau)
   - 🔒 **TUTUP** (background merah)
9. **Lihat confidence score** (persentase keyakinan model)
10. **Expand "Detail Fitur"** untuk melihat 29 fitur yang diekstrak

### Metode 2: Input Manual Fitur (Advanced)

1. **Pilih tab "Input Manual Fitur"**
2. **Masukkan nilai untuk 29 fitur**:
   - MFCC 1-13 (mean & std)
   - Spectral Centroid
   - Spectral Bandwidth
   - Zero Crossing Rate
3. **Klik "Prediksi dari Fitur Manual"**
4. **Lihat hasil prediksi**

### Tips Penggunaan Audio

✅ **DO's:**
- Gunakan audio berkualitas baik
- Pastikan suara jelas terdengar
- Durasi ideal: 1-3 detik
- Minimal background noise
- Format WAV atau MP3

❌ **DON'Ts:**
- Jangan gunakan audio yang terlalu pendek (< 0.5 detik)
- Hindari audio dengan banyak noise
- Jangan gunakan audio dengan volume terlalu rendah
- Hindari audio dengan multiple speakers

## 💻 Untuk Developer

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd project_2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
```

### Testing

```bash
# Test aplikasi
python test_app.py

# Test model files
python check_metadata.py
```

### Modifikasi

#### Menambah Fitur Audio
Edit fungsi `extract_audio_features()` di `app.py`:
```python
def extract_audio_features(audio_data, sr=22050, n_features=29):
    # Tambah ekstraksi fitur baru di sini
    # Pastikan total fitur tetap 29
    pass
```

#### Mengubah Tampilan
Edit bagian Streamlit markdown di `app.py`:
```python
st.markdown("""
    <div style='...'>
        <!-- Custom HTML/CSS di sini -->
    </div>
""", unsafe_allow_html=True)
```

#### Update Model
1. Train model baru dengan 29 fitur
2. Save sebagai `audio_classifier.pkl`
3. Update metadata
4. Test dengan `test_app.py`
5. Deploy

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "File not found"
- Pastikan semua file .pkl ada di direktori yang sama
- Check current directory dengan `os.getcwd()`

### Audio tidak ter-load
- Cek format file (harus WAV, MP3, OGG, FLAC)
- Cek ukuran file (< 200MB)
- Install soundfile: `pip install soundfile`

### Prediksi selalu sama
- Check apakah model ter-load dengan benar
- Verify dengan `python test_app.py`
- Check ekstraksi fitur menghasilkan nilai berbeda

### Aplikasi lambat
- Reduce audio sampling rate
- Optimize feature extraction
- Use caching: `@st.cache_data`

## 📞 Support

Jika menemukan bug atau punya pertanyaan:
1. Check README.md
2. Check DEPLOYMENT.md
3. Run test_app.py untuk diagnosis
4. Contact developer team

## 🎯 Best Practices

### Untuk Akurasi Terbaik:
1. **Kualitas Audio**: Gunakan recording yang jelas
2. **Format**: WAV 16-bit, 22050 Hz
3. **Durasi**: 1-3 detik optimal
4. **Environment**: Ruangan tenang, minimal echo
5. **Pronunciation**: Ucapkan kata dengan jelas

### Untuk Performance:
1. Compress audio jika > 5MB
2. Use mono channel (bukan stereo)
3. Sample rate 22050 Hz (default librosa)

## 📊 Understanding Results

### Confidence Score
- **90-100%**: Very High Confidence ⭐⭐⭐⭐⭐
- **80-90%**: High Confidence ⭐⭐⭐⭐
- **70-80%**: Good Confidence ⭐⭐⭐
- **60-70%**: Moderate Confidence ⭐⭐
- **< 60%**: Low Confidence ⭐

### Ketika Confidence Rendah
1. Check kualitas audio
2. Verify bahwa kata diucapkan dengan benar
3. Try re-record dengan kondisi lebih baik
4. Check untuk background noise

## 🚀 Advanced Usage

### Batch Processing
```python
import librosa
import pickle
import numpy as np

# Load model
with open('audio_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Process multiple files
for file in audio_files:
    audio, sr = librosa.load(file, sr=22050)
    features = extract_audio_features(audio, sr)
    prediction = model.predict(features.reshape(1, -1))
    print(f"{file}: {prediction}")
```

### API Integration
Aplikasi bisa di-wrap sebagai API menggunakan FastAPI/Flask untuk integrasi dengan sistem lain.

---

**Happy Coding! 🎉**
