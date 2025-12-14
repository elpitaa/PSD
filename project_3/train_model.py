"""
Script untuk training model 1D CNN dan menyimpan ke project_3
"""
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Training Model 1D CNN untuk Arabic Digit Recognition")
print("="*60)

# Load dataset
print("\n1. Loading dataset...")

def parse_file(filepath, blocks_per_digit):
    """Parse file dan ekstrak features"""
    sequences = []
    current_sequence = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Empty line = end of utterance
            if not line:
                if len(current_sequence) > 0:
                    sequences.append(np.array(current_sequence))
                    current_sequence = []
                continue
            
            # Parse MFCC features (13 values)
            try:
                values = [float(x) for x in line.split()]
                if len(values) == 13:
                    current_sequence.append(values)
            except:
                pass
    
    # Add last sequence
    if len(current_sequence) > 0:
        sequences.append(np.array(current_sequence))
    
    # Generate labels based on block position
    # 660 blocks per digit (train) or 220 (test)
    labels = []
    for digit in range(10):
        labels.extend([digit] * blocks_per_digit)
    
    labels = np.array(labels[:len(sequences)])
    
    return sequences, labels

train_file = '../tugas/data/Train_Arabic_Digit.txt'
test_file = '../tugas/data/Test_Arabic_Digit.txt'

X_train_raw, y_train = parse_file(train_file, blocks_per_digit=660)
X_test_raw, y_test = parse_file(test_file, blocks_per_digit=220)

print(f"   Train samples: {len(X_train_raw)}")
print(f"   Test samples: {len(X_test_raw)}")

# Padding ke panjang yang sama
print("\n2. Padding sequences...")
max_length = 93

def pad_sequences(X, max_len):
    """Padding atau truncate ke max_len"""
    X_padded = []
    for x in X:
        if x.shape[0] < max_len:
            # Padding
            pad_width = max_len - x.shape[0]
            x_padded = np.pad(x, ((0, pad_width), (0, 0)), mode='constant')
        else:
            # Truncate
            x_padded = x[:max_len, :]
        X_padded.append(x_padded)
    return np.array(X_padded)

X_train = pad_sequences(X_train_raw, max_length)
X_test = pad_sequences(X_test_raw, max_length)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Normalisasi dengan StandardScaler
print("\n3. Normalization...")
scaler = StandardScaler()

# Reshape untuk scaler (samples, features)
X_train_flat = X_train.reshape(-1, X_train.shape[2])
scaler.fit(X_train_flat)

# Transform train dan test
X_train_normalized = scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test_normalized = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

print(f"   Scaler fitted on {X_train_flat.shape[0]} samples")

# Check class distribution
print("\nClass distribution in training set:")
for digit in range(10):
    count = np.sum(y_train == digit)
    print(f"  Digit {digit}: {count} samples ({count/len(y_train)*100:.1f}%)")

# Build 1D CNN model (Simplified and better regularization)
print("\n4. Building 1D CNN model...")
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(max_length, 13)),
    
    # Layer 1: Conv1D block
    layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.25),
    
    # Layer 2: Conv1D block  
    layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.25),
    
    # Layer 3: Conv1D block
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    
    # Dense layers
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   Model Summary:")
model.summary()

# Training
print("\n5. Training model...")
print("   This may take a few minutes...")

# Stratified validation split untuk memastikan distribusi seimbang
from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_normalized, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

print(f"   Train: {len(X_train_split)} samples, Validation: {len(X_val)} samples")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,  # Increase patience
    restore_best_weights=True,
    mode='max'
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,  # Increase patience
    min_lr=1e-7,
    mode='max'
)

history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),  # Gunakan stratified validation
    epochs=100,  # Increase epochs
    batch_size=32,  # Smaller batch size
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluation
print("\n6. Evaluating model...")
y_train_pred = np.argmax(model.predict(X_train_split, verbose=0), axis=1)  # Use split data
y_test_pred = np.argmax(model.predict(X_test_normalized, verbose=0), axis=1)

train_accuracy = accuracy_score(y_train_split, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n   Train Accuracy: {train_accuracy*100:.2f}%")
print(f"   Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"   Test F1-Score:  {test_f1:.4f}")

# Evaluasi per digit
print("\n   Per-digit Test Accuracy:")
for digit in range(10):
    mask = y_test == digit
    if np.sum(mask) > 0:
        acc = accuracy_score(y_test[mask], y_test_pred[mask])
        print(f"      Digit {digit}: {acc*100:.2f}% ({np.sum(mask)} samples)")

# Save model dan scaler
print("\n7. Saving model and scaler...")

# Save model (format .h5)
model.save('best_model.h5')
print(f"   ✓ Model saved to: best_model.h5")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ Scaler saved to: scaler.pkl")

# Save metadata
metadata = {
    'model_name': '1D CNN',
    'model_path': 'best_model.h5',
    'scaler_path': 'scaler.pkl',
    'test_accuracy': float(test_accuracy),
    'test_f1_score': float(test_f1),
    'num_classes': 10,
    'max_length': max_length,
    'num_mfcc': 13,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'class_names': [str(i) for i in range(10)]
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, indent=2, fp=f)
print(f"   ✓ Metadata saved to: model_metadata.json")

print("\n" + "="*60)
print("✓ Training completed successfully!")
print("="*60)
print(f"\nModel files saved in: /workspaces/PSD/project_3/")
print(f"  - best_model.h5")
print(f"  - scaler.pkl")
print(f"  - model_metadata.json")
print(f"\nNow you can run the Streamlit app!")
