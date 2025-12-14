"""
Debug audio file yang diupload ke Streamlit
Untuk cek apakah preprocessing bekerja dengan benar
"""
import numpy as np
import librosa
import pickle
from tensorflow import keras
import sys
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) < 2:
    print("Usage: python debug_audio.py <audio_file_path>")
    print("Example: python debug_audio.py wahid.m4a")
    sys.exit(1)

audio_path = sys.argv[1]

print("="*60)
print("DEBUG AUDIO FILE")
print("="*60)

# Load model dan scaler
print("\n1. Loading model dan scaler...")
model = keras.models.load_model('best_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load metadata
import json
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)
    
print(f"   Model accuracy: {metadata['test_accuracy']*100:.2f}%")
print(f"   Training date: {metadata['training_date']}")

# Load audio (SAMA SEPERTI DI APP.PY)
print(f"\n2. Loading audio: {audio_path}")
audio_data, sr = librosa.load(audio_path, sr=11025)
print(f"   Sampling rate: {sr} Hz")
print(f"   Audio length: {len(audio_data)/sr:.2f} seconds")
print(f"   Audio shape: {audio_data.shape}")

# Extract MFCC (SAMA SEPERTI DI APP.PY)
print("\n3. Extracting MFCC features...")
n_mfcc = metadata['num_mfcc']
max_length = metadata['max_length']

mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
print(f"   MFCC shape before padding: {mfcc.shape}")

# Padding atau truncate
if mfcc.shape[1] < max_length:
    pad_width = max_length - mfcc.shape[1]
    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
else:
    mfcc = mfcc[:, :max_length]

mfcc_features = mfcc.T  # Transpose: (max_length, n_mfcc)
print(f"   MFCC shape after padding: {mfcc_features.shape}")
print(f"   MFCC mean: {mfcc_features.mean():.4f}")
print(f"   MFCC std: {mfcc_features.std():.4f}")

# Normalize (SAMA SEPERTI DI APP.PY)
print("\n4. Normalizing features...")
normalized_features = scaler.transform(mfcc_features)
print(f"   Normalized mean: {normalized_features.mean():.4f}")
print(f"   Normalized std: {normalized_features.std():.4f}")

# Reshape untuk model
features = normalized_features.reshape(1, normalized_features.shape[0], normalized_features.shape[1])
print(f"   Input shape: {features.shape}")

# Predict
print("\n5. Predicting...")
predictions = model.predict(features, verbose=0)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)
print(f"Predicted Digit: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
print("\nAll probabilities:")
for i, prob in enumerate(predictions[0]):
    bar = "█" * int(prob * 50)
    print(f"  Digit {i}: {prob*100:5.2f}% {bar}")

print("\n" + "="*60)
print("DIAGNOSIS:")
if confidence > 0.9:
    print("✓ High confidence - Model yakin dengan prediksi")
else:
    print("⚠ Low confidence - Audio mungkin tidak sesuai training data")
    
if predicted_class == 6 and confidence < 0.7:
    print("⚠ Prediksi 6 dengan confidence rendah")
    print("  → Audio kemungkinan tidak dari dataset Arabic Digits")
    print("  → Atau kualitas audio berbeda dari training data")
