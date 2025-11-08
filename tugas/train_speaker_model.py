#!/usr/bin/env python3
"""
Script untuk training model Speaker Identification
Generate file .pkl baru untuk aplikasi Streamlit
"""

print("="*70)
print("🚀 TRAINING MODEL - SPEAKER IDENTIFICATION")
print("="*70)

# Import libraries
import numpy as np
import pandas as pd
import os
from glob import glob
from datetime import datetime
import librosa
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("\n✅ Libraries imported successfully!")

# ===== FUNCTIONS =====

def load_audio(file_path, sr=22050, duration=None):
    """Load audio file"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        return audio, sr
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {str(e)}")
        return None, None

def extract_comprehensive_features(audio, sr):
    """Extract ~100 features untuk speaker identification"""
    features = {}
    
    # TIME DOMAIN FEATURES
    features['mean'] = np.mean(audio)
    features['std'] = np.std(audio)
    features['max'] = np.max(audio)
    features['min'] = np.min(audio)
    features['median'] = np.median(audio)
    features['variance'] = np.var(audio)
    features['skewness'] = stats.skew(audio)
    features['kurtosis'] = stats.kurtosis(audio)
    features['range'] = np.ptp(audio)
    features['iqr'] = stats.iqr(audio)
    features['energy'] = np.sum(audio**2)
    features['rms'] = np.sqrt(np.mean(audio**2))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    features['duration'] = len(audio) / sr
    
    gradient = np.gradient(audio)
    features['mean_gradient'] = np.mean(np.abs(gradient))
    features['max_gradient'] = np.max(np.abs(gradient))
    features['percentile_25'] = np.percentile(audio, 25)
    features['percentile_75'] = np.percentile(audio, 75)
    
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    features['autocorr_max'] = np.max(autocorr[1:100]) / autocorr[0] if autocorr[0] != 0 else 0
    
    # SPECTRAL FEATURES
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_centroid_max'] = np.max(spectral_centroids)
    features['spectral_centroid_min'] = np.min(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast)
    features['spectral_contrast_std'] = np.std(spectral_contrast)
    
    # MFCC (20 coefficients - CRITICAL for Speaker ID)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
    
    # PITCH/F0
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            features['pitch_mean'] = np.mean(f0_clean)
            features['pitch_std'] = np.std(f0_clean)
            features['pitch_max'] = np.max(f0_clean)
            features['pitch_min'] = np.min(f0_clean)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_max'] = 0
            features['pitch_min'] = 0
            features['pitch_range'] = 0
    except:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_max'] = 0
        features['pitch_min'] = 0
        features['pitch_range'] = 0
    
    # CHROMA
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    return features

def create_speaker_dataset(base_path='audio_data2', sr=22050, max_duration=5):
    """Create dataset dengan speaker identification"""
    data = []
    
    print("\n" + "="*70)
    print("🎵 LOADING & EXTRACTING FEATURES")
    print("="*70)
    
    # NADIA
    nadia_path = os.path.join(base_path, 'BukaTutup_nadia')
    print(f"\n📁 Processing NADIA's audio...")
    
    if os.path.exists(nadia_path):
        # Nadia - Buka
        nadia_buka_files = glob(os.path.join(nadia_path, 'buka*.mp3'))
        print(f"   → Buka: {len(nadia_buka_files)} files")
        
        for i, file_path in enumerate(nadia_buka_files):
            audio, sr_loaded = load_audio(file_path, sr=sr, duration=max_duration)
            if audio is not None:
                features = extract_comprehensive_features(audio, sr_loaded)
                features['label'] = 'nadia_buka'
                features['speaker'] = 'nadia'
                features['action'] = 'buka'
                features['filename'] = os.path.basename(file_path)
                data.append(features)
            
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i + 1}/{len(nadia_buka_files)}")
        
        # Nadia - Tutup
        nadia_tutup_files = glob(os.path.join(nadia_path, 'tutup*.mp3'))
        print(f"   → Tutup: {len(nadia_tutup_files)} files")
        
        for i, file_path in enumerate(nadia_tutup_files):
            audio, sr_loaded = load_audio(file_path, sr=sr, duration=max_duration)
            if audio is not None:
                features = extract_comprehensive_features(audio, sr_loaded)
                features['label'] = 'nadia_tutup'
                features['speaker'] = 'nadia'
                features['action'] = 'tutup'
                features['filename'] = os.path.basename(file_path)
                data.append(features)
            
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i + 1}/{len(nadia_tutup_files)}")
    
    # UFI
    ufi_path = os.path.join(base_path, 'BukaTutup_ufi/Rekaman')
    print(f"\n📁 Processing UFI's audio...")
    
    if os.path.exists(ufi_path):
        # Ufi - Buka
        ufi_buka_path = os.path.join(ufi_path, 'Buka')
        ufi_buka_files = glob(os.path.join(ufi_buka_path, '*.wav')) + glob(os.path.join(ufi_buka_path, '*.mp3'))
        print(f"   → Buka: {len(ufi_buka_files)} files")
        
        for i, file_path in enumerate(ufi_buka_files):
            audio, sr_loaded = load_audio(file_path, sr=sr, duration=max_duration)
            if audio is not None:
                features = extract_comprehensive_features(audio, sr_loaded)
                features['label'] = 'ufi_buka'
                features['speaker'] = 'ufi'
                features['action'] = 'buka'
                features['filename'] = os.path.basename(file_path)
                data.append(features)
            
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i + 1}/{len(ufi_buka_files)}")
        
        # Ufi - Tutup
        ufi_tutup_path = os.path.join(ufi_path, 'tutup')
        ufi_tutup_files = glob(os.path.join(ufi_tutup_path, '*.wav')) + glob(os.path.join(ufi_tutup_path, '*.mp3'))
        print(f"   → Tutup: {len(ufi_tutup_files)} files")
        
        for i, file_path in enumerate(ufi_tutup_files):
            audio, sr_loaded = load_audio(file_path, sr=sr, duration=max_duration)
            if audio is not None:
                features = extract_comprehensive_features(audio, sr_loaded)
                features['label'] = 'ufi_tutup'
                features['speaker'] = 'ufi'
                features['action'] = 'tutup'
                features['filename'] = os.path.basename(file_path)
                data.append(features)
            
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i + 1}/{len(ufi_tutup_files)}")
    
    if len(data) == 0:
        print("❌ No data processed!")
        return None
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*70)
    print("✅ DATASET CREATED!")
    print("="*70)
    print(f"\n📊 Total samples: {len(df)}")
    print(f"📋 Label distribution:\n{df['label'].value_counts()}")
    
    return df

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Create dataset
    df = create_speaker_dataset('audio_data2', sr=22050, max_duration=5)
    
    if df is None:
        print("❌ Failed to create dataset!")
        exit(1)
    
    # Save dataset
    df.to_csv('speaker_audio_features.csv', index=False)
    print(f"\n💾 Dataset saved: speaker_audio_features.csv")
    
    # Prepare data
    print("\n" + "="*70)
    print("⚙️ PREPROCESSING DATA")
    print("="*70)
    
    metadata_cols = ['label', 'speaker', 'action', 'filename']
    X = df.drop(columns=metadata_cols, errors='ignore')
    y = df['label']
    
    feature_names = X.columns.tolist()
    
    # Handle missing/inf
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Train model
    print("\n" + "="*70)
    print("🤖 TRAINING MODEL - Random Forest")
    print("="*70)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model
    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/speaker_model.pkl')
    joblib.dump(scaler, 'models/speaker_model_scaler.pkl')
    joblib.dump(label_encoder, 'models/speaker_model_label_encoder.pkl')
    joblib.dump(feature_names, 'models/speaker_model_feature_names.pkl')
    
    metadata = {
        'model_type': 'RandomForestClassifier',
        'test_accuracy': accuracy,
        'cv_score': 0.0,
        'best_params': {},
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'classes': list(label_encoder.classes_),
        'n_classes': len(label_encoder.classes_),
        'speakers': ['nadia', 'ufi'],
        'actions': ['buka', 'tutup'],
        'sampling_rate': 22050,
        'purpose': 'Speaker Identification & Action Classification',
        'n_features': len(feature_names),
        'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(metadata, 'models/speaker_model_metadata.pkl')
    
    print("\n✅ Model files saved:")
    print("   ├── speaker_model.pkl")
    print("   ├── speaker_model_scaler.pkl")
    print("   ├── speaker_model_label_encoder.pkl")
    print("   ├── speaker_model_feature_names.pkl")
    print("   └── speaker_model_metadata.pkl")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print(f"📊 Accuracy: {accuracy*100:.2f}%")
    print(f"👥 Speakers: {', '.join(metadata['speakers'])}")
    print(f"🎬 Actions: {', '.join(metadata['actions'])}")
    print(f"📁 Location: models/")
    print("\n✅ Ready for deployment!")
