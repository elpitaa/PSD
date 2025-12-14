"""
Test audio file secara langsung (tanpa Streamlit)
Untuk debug kenapa prediksi selalu 6
"""
import numpy as np
import librosa
import pickle
from tensorflow import keras
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TEST AUDIO FILE - DEBUG PREDIKSI")
print("="*70)

# Load model dan scaler
print("\n1. Loading model dan scaler...")
model = keras.models.load_model('best_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Test dengan sample dari TEST DATA dulu (sebagai baseline)
print("\n2. BASELINE TEST - Sample dari test data:")
print("-" * 70)

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

X_test_raw, y_test = parse_file('../tugas/data/Test_Arabic_Digit.txt', blocks_per_digit=220)

# Pad
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

# Test digit 1 (wahid) dari test data
digit_1_idx = 220  # Index pertama digit 1
sample_1 = X_test[digit_1_idx]

normalized = scaler.transform(sample_1)
features = normalized.reshape(1, 93, 13)
predictions = model.predict(features, verbose=0)
pred_digit = np.argmax(predictions[0])

print(f"Test data digit 1: Predicted={pred_digit}, Confidence={predictions[0][pred_digit]*100:.1f}%")
print(f"  Top 3 predictions:")
top3 = np.argsort(predictions[0])[-3:][::-1]
for i in top3:
    print(f"    Digit {i}: {predictions[0][i]*100:.1f}%")

# Statistik MFCC
print(f"\n  MFCC stats (test data):")
print(f"    Mean: {sample_1.mean():.4f}")
print(f"    Std: {sample_1.std():.4f}")
print(f"    Min: {sample_1.min():.4f}")
print(f"    Max: {sample_1.max():.4f}")

print("\n" + "="*70)
print("KESIMPULAN:")
print("="*70)
print("\nJika test data digit 1 diprediksi BENAR:")
print("  → Model bekerja dengan baik")
print("  → Audio file yang kamu upload BERBEDA dari training data")
print("\nPenyebab audio berbeda:")
print("  1. Bukan native Arabic speaker")
print("  2. Aksen/intonasi berbeda")
print("  3. Kualitas rekaman berbeda (noise, volume, dll)")
print("  4. Background audio berbeda")
print("  5. Format audio berbeda (sample rate, encoding)")
print("\nSOLUSI:")
print("  → Model hanya akurat untuk audio MIRIP dengan training data")
print("  → Untuk audio umum, perlu retrain dengan dataset lebih beragam")
print("  → Atau tambahkan data augmentation saat training")
