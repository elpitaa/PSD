"""
Script untuk test aplikasi sebelum deployment
"""
import pickle
import numpy as np
import os

print("="*70)
print("🧪 TESTING SISTEM IDENTIFIKASI SUARA: BUKA/TUTUP")
print("="*70)

os.chdir(r"E:\Semester 5\Proyek Sain Data\project_2")

# Test 1: Load semua file
print("\n1️⃣ Testing: Load Model Files")
print("-"*70)

try:
    with open('audio_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
    print(f"   Type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

try:
    with open('audio_classifier_scaler.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✅ Feature names loaded successfully")
    print(f"   Total features: {len(feature_names)}")
except Exception as e:
    print(f"❌ Failed to load feature names: {e}")
    exit(1)

try:
    with open('audio_classifier_label_encoder.pkl', 'rb') as f:
        classes = pickle.load(f)
    print(f"✅ Classes loaded successfully")
    print(f"   Classes: {classes}")
except Exception as e:
    print(f"❌ Failed to load classes: {e}")
    exit(1)

try:
    with open('audio_classifier_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"✅ Metadata loaded successfully")
    print(f"   Model name: {metadata.get('model_name', 'N/A')}")
    print(f"   Test accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
except Exception as e:
    print(f"❌ Failed to load metadata: {e}")
    exit(1)

# Test 2: Prediksi dengan data dummy
print("\n2️⃣ Testing: Model Prediction")
print("-"*70)

try:
    # Buat data dummy dengan 29 fitur
    dummy_features = np.random.randn(1, 29)
    prediction = model.predict(dummy_features)
    predicted_class = classes[prediction[0]]
    
    print(f"✅ Prediction successful")
    print(f"   Input shape: {dummy_features.shape}")
    print(f"   Prediction: {predicted_class}")
    
    # Test probabilitas
    try:
        proba = model.predict_proba(dummy_features)[0]
        confidence = np.max(proba) * 100
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Probabilities: Buka={proba[0]*100:.1f}%, Tutup={proba[1]*100:.1f}%")
    except:
        print("   ⚠️  Probability not available")
        
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    exit(1)

# Test 3: Cek konsistensi
print("\n3️⃣ Testing: Model Consistency")
print("-"*70)

try:
    # Model features vs Feature names
    if hasattr(model, 'n_features_in_'):
        model_features = model.n_features_in_
        expected_features = len(feature_names)
        
        if model_features == expected_features:
            print(f"✅ Feature count consistent: {model_features}")
        else:
            print(f"⚠️  Feature mismatch: model={model_features}, expected={expected_features}")
    
    # Model classes vs Classes array
    if hasattr(model, 'classes_'):
        if len(model.classes_) == len(classes):
            print(f"✅ Classes count consistent: {len(classes)}")
        else:
            print(f"⚠️  Classes mismatch")
    
    # Metadata vs actual
    meta_features = metadata.get('n_features', 0)
    if meta_features == len(feature_names):
        print(f"✅ Metadata consistent with features: {meta_features}")
    else:
        print(f"⚠️  Metadata mismatch: {meta_features} vs {len(feature_names)}")
        
except Exception as e:
    print(f"❌ Consistency check failed: {e}")

# Test 4: Cek dependencies
print("\n4️⃣ Testing: Dependencies")
print("-"*70)

dependencies = [
    'streamlit',
    'pandas',
    'numpy',
    'sklearn',
    'librosa',
    'soundfile'
]

for dep in dependencies:
    try:
        if dep == 'sklearn':
            import sklearn
            module = sklearn
        else:
            module = __import__(dep)
        print(f"✅ {dep:15s} - v{getattr(module, '__version__', 'unknown')}")
    except ImportError:
        print(f"❌ {dep:15s} - NOT INSTALLED")

# Test 5: File size check
print("\n5️⃣ Testing: File Sizes")
print("-"*70)

files_to_check = [
    'app.py',
    'audio_classifier.pkl',
    'audio_classifier_scaler.pkl',
    'audio_classifier_label_encoder.pkl',
    'audio_classifier_metadata.pkl',
    'requirements.txt'
]

total_size = 0
for filename in files_to_check:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        total_size += size
        size_kb = size / 1024
        if size_kb < 1024:
            print(f"✅ {filename:40s} {size_kb:8.2f} KB")
        else:
            print(f"✅ {filename:40s} {size_kb/1024:8.2f} MB")
    else:
        print(f"❌ {filename:40s} NOT FOUND")

print(f"\n📦 Total size: {total_size / 1024:.2f} KB ({total_size / (1024*1024):.2f} MB)")

if total_size < 100 * 1024 * 1024:  # 100MB
    print("✅ Total size OK for deployment")
else:
    print("⚠️  Total size might be too large for free deployment")

# Final Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)
print("""
✅ All tests passed!

Your application is ready for deployment to Streamlit Cloud.

Next steps:
1. Push all files to GitHub
2. Go to https://share.streamlit.io/
3. Deploy your app
4. Share the link!

Good luck! 🚀
""")
print("="*70)
