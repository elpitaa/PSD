"""
Verify bahwa pipeline di app.py menghasilkan prediksi yang sama dengan test_model.py
Test menggunakan data test yang sama
"""
import numpy as np
import pickle
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("VERIFIKASI PIPELINE STREAMLIT vs TEST MODEL")
print("="*60)

# Load model dan scaler (sama seperti di app.py)
print("\n1. Loading model dan scaler...")
model = keras.models.load_model('best_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data (sama seperti di train_model.py)
print("2. Loading test data...")
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

# Padding (sama seperti extract_mfcc di app.py)
print("3. Padding sequences...")
max_length = 93

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

X_test = pad_sequences(X_test_raw, max_length)

# Test dengan 10 sample (1 per digit)
print("\n4. Testing 10 samples (1 per digit)...")
print("-" * 60)

for digit in range(10):
    # Ambil sample pertama dari digit ini
    idx = digit * 220  # 220 samples per digit
    sample = X_test[idx]
    true_label = y_test[idx]
    
    # SIMULATE APP.PY PIPELINE
    # Step 1: Normalize (sama seperti predict_digit di app.py)
    normalized = scaler.transform(sample)
    
    # Step 2: Reshape (sama seperti predict_digit di app.py)
    features = normalized.reshape(1, normalized.shape[0], normalized.shape[1])
    
    # Step 3: Predict
    predictions = model.predict(features, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Display result
    status = "✓" if predicted_class == true_label else "✗"
    print(f"{status} Digit {digit}: Predicted={predicted_class}, True={true_label}, Confidence={confidence*100:.1f}%")

print("-" * 60)
print("\n✓ Pipeline verification complete!")
print("\nKesimpulan:")
print("- Jika semua ✓ → Pipeline bekerja dengan benar")
print("- Jika ada ✗ → Ada masalah di model atau preprocessing")
print("\nJika pipeline benar tapi Streamlit prediksi salah:")
print("→ Audio yang diupload BUKAN dari dataset Arabic Digits")
print("→ Model tidak bisa prediksi audio random/bahasa lain")
