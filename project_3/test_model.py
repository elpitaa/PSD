"""
Test model dengan data training untuk memastikan model bekerja
"""
import numpy as np
import pickle
from tensorflow import keras
import os

print("Loading model dan scaler...")
model = keras.models.load_model('best_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load beberapa sample dari training data
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
    
    labels = np.array(labels[:len(sequences)])
    return sequences, labels

print("\nLoading test data...")
X_test_raw, y_test = parse_file('../tugas/data/Test_Arabic_Digit.txt', blocks_per_digit=220)

# Padding
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
print(f"X_test shape: {X_test.shape}")

# Normalisasi
X_test_normalized = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
print(f"Normalized")

# Test beberapa sample
print("\n" + "="*60)
print("Testing 10 samples dari setiap digit (total 100 samples)")
print("="*60)

for digit in range(10):
    # Ambil 10 sample dari digit ini
    start_idx = digit * 220
    end_idx = start_idx + 10
    
    X_sample = X_test_normalized[start_idx:end_idx]
    y_sample = y_test[start_idx:end_idx]
    
    # Predict
    predictions = model.predict(X_sample, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Hitung akurasi
    correct = np.sum(y_pred == y_sample)
    accuracy = correct / len(y_sample) * 100
    
    print(f"\nDigit {digit}:")
    print(f"  True labels: {y_sample.tolist()}")
    print(f"  Predictions: {y_pred.tolist()}")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/10 correct)")
    
    # Tampilkan confidence untuk sample pertama
    print(f"  Confidence (sample 0): {predictions[0][y_pred[0]]*100:.1f}%")
    print(f"  All probabilities (sample 0): {predictions[0]}")

print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)

# Predict semua
all_predictions = model.predict(X_test_normalized, verbose=0)
all_pred_classes = np.argmax(all_predictions, axis=1)

# Confusion per digit
from collections import Counter
print("\nPrediction distribution:")
pred_counter = Counter(all_pred_classes)
for digit in range(10):
    count = pred_counter.get(digit, 0)
    percentage = count / len(all_pred_classes) * 100
    print(f"  Digit {digit}: {count:4d} samples ({percentage:5.1f}%)")

# Overall accuracy
overall_accuracy = np.sum(all_pred_classes == y_test) / len(y_test) * 100
print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")
