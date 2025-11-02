"""
Script untuk melihat isi metadata dan struktur model
"""
import pickle
import os
import numpy as np

os.chdir(r"E:\Semester 5\Proyek Sain Data\project_2")

print("="*60)
print("CHECKING MODEL FILES")
print("="*60)

# Load metadata
with open('audio_classifier_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("\n📋 METADATA CONTENT:")
print("-"*60)
for key, value in metadata.items():
    print(f"{key}: {value}")

# Load label encoder
with open('audio_classifier_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("\n🏷️ LABEL ENCODER:")
print("-"*60)
print(f"Type: {type(label_encoder)}")
print(f"Classes: {label_encoder}")

# Load scaler
with open('audio_classifier_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("\n⚖️ SCALER:")
print("-"*60)
print(f"Type: {type(scaler)}")
if hasattr(scaler, 'shape'):
    print(f"Shape: {scaler.shape}")
print(f"Data preview: {scaler[:5] if len(scaler) > 5 else scaler}")

# Load model
with open('audio_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

print("\n🤖 MODEL:")
print("-"*60)
print(f"Type: {type(model)}")
print(f"Model info: {model}")

if hasattr(model, 'classes_'):
    print(f"Classes: {model.classes_}")
if hasattr(model, 'n_features_in_'):
    print(f"Number of features: {model.n_features_in_}")

print("\n" + "="*60)
