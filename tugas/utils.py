"""
Utility Functions untuk Arabic Digit Classification
Berisi helper functions untuk data loading, preprocessing, dan visualisasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os


def load_train_test_data(data_dir):
    """
    Load dataset dari file Train_Arabic_Digit.txt dan Test_Arabic_Digit.txt
    
    Format file:
    - Lines dengan 13 nilai = MFCC features (1 frame)
    - Baris kosong = separator antar utterances
    - Setiap utterance = 1 spoken digit (4-93 frames)
    - Train: 660 utterances per digit (digit 0-9 berurutan)
    - Test: 220 utterances per digit (digit 0-9 berurutan)
    
    Parameters:
    - data_dir: path ke folder data
    
    Returns:
    - X_train: training features (n_samples, max_length, 13)
    - y_train: training labels
    - X_test: test features (n_samples, max_length, 13)
    - y_test: test labels
    - max_length: maximum sequence length
    """
    
    def parse_file(filepath, blocks_per_digit):
        """Parse file dan ekstrak features, infer labels dari posisi block"""
        sequences = []
        current_sequence = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Empty or whitespace-only line = end of utterance
                if not line:
                    if len(current_sequence) > 0:
                        sequences.append(np.array(current_sequence))
                        current_sequence = []
                    continue
                
                # Try to parse as MFCC features (13 values)
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == 13:
                        current_sequence.append(values)
                except:
                    # Skip lines that can't be parsed
                    pass
        
        # Add last sequence if exists
        if len(current_sequence) > 0:
            sequences.append(np.array(current_sequence))
        
        # Generate labels based on block position
        # Blocks are organized: digit 0 (blocks 0-blocks_per_digit-1), digit 1 (blocks_per_digit-2*blocks_per_digit-1), etc.
        num_digits = 10
        labels = []
        for digit in range(num_digits):
            labels.extend([digit] * blocks_per_digit)
        
        labels = np.array(labels[:len(sequences)])  # Trim to actual number of sequences
        
        return sequences, labels
    
    # Load train and test data
    train_file = os.path.join(data_dir, 'Train_Arabic_Digit.txt')
    test_file = os.path.join(data_dir, 'Test_Arabic_Digit.txt')
    
    print(f"Loading training data from {train_file}...")
    X_train_list, y_train = parse_file(train_file, blocks_per_digit=660)
    
    print(f"Loading test data from {test_file}...")
    X_test_list, y_test = parse_file(test_file, blocks_per_digit=220)
    
    # Find max sequence length
    max_length = max([len(seq) for seq in X_train_list + X_test_list])
    print(f"Maximum sequence length: {max_length}")
    
    # Pad sequences to max_length
    def pad_sequences(sequences, max_len, n_features=13):
        """Pad sequences to max_len"""
        padded = np.zeros((len(sequences), max_len, n_features))
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            padded[i, :seq_len, :] = seq[:seq_len]
        return padded
    
    X_train = pad_sequences(X_train_list, max_length)
    X_test = pad_sequences(X_test_list, max_length)
    
    print(f"\nDataset loaded successfully!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training labels: {len(y_train)} (digits 0-9: {np.bincount(y_train)})")
    print(f"Test labels: {len(y_test)} (digits 0-9: {np.bincount(y_test)})")
    
    return X_train, y_train, X_test, y_test, max_length


def plot_class_distribution(y_train, y_test):
    """Plot distribusi kelas untuk train dan test set"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training distribution
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0].bar(unique, counts, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Digit', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(unique)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Test distribution
    unique, counts = np.unique(y_test, return_counts=True)
    axes[1].bar(unique, counts, color='coral', alpha=0.8)
    axes[1].set_xlabel('Digit', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(unique)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Dataset is balanced across all classes")


def plot_sample_mfcc(X_data, y_data, num_samples=5):
    """Visualisasi sample MFCC features"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(X_data))
        sample = X_data[idx]
        label = y_data[idx]
        
        # Find actual length (remove padding)
        non_zero_rows = np.any(sample != 0, axis=1)
        actual_length = np.sum(non_zero_rows)
        
        # Plot MFCC heatmap
        im = axes[i].imshow(sample[:actual_length].T, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_xlabel('Time Frames', fontsize=11)
        axes[i].set_ylabel('MFCC Coefficients', fontsize=11)
        axes[i].set_title(f'Sample {i+1}: Digit {label} (Length: {actual_length} frames)', 
                         fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()


def normalize_data(X_train, X_test, scaler_path='../project_3/scaler.pkl'):
    """
    Normalize MFCC features using StandardScaler
    
    Parameters:
    - X_train: training data
    - X_test: test data
    - scaler_path: path to save the scaler (relative from /workspaces/PSD/tugas/)
    
    Returns:
    - X_train_norm: normalized training data
    - X_test_norm: normalized test data
    - scaler: fitted StandardScaler object
    """
    # Reshape to 2D for scaling
    n_train, timesteps, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back to 3D
    X_train_norm = X_train_scaled.reshape(n_train, timesteps, n_features)
    X_test_norm = X_test_scaled.reshape(n_test, timesteps, n_features)
    
    # Save scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    return X_train_norm, X_test_norm, scaler


def create_validation_split(X_train, y_train, val_split=0.15, random_state=42):
    """Create validation split from training data"""
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_split, 
        random_state=random_state,
        stratify=y_train
    )
    return X_train_split, X_val, y_train_split, y_val


def plot_training_history(history):
    """Plot training history (loss and accuracy)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_models(results_dict):
    """
    Compare multiple models
    
    Parameters:
    - results_dict: dict with format {'Model Name': {'Accuracy': val, 'F1-Score': val}}
    """
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['Accuracy'] for m in models]
    f1_scores = [results_dict[m]['F1-Score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral', alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    df = pd.DataFrame(results_dict).T
    print(df.to_string())
    print("="*60)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report"""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    print("="*60)
