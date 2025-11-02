import pickle
import os

print("Current directory:", os.getcwd())
print("\nTesting pickle files...")

files = [
    "audio_classifier.pkl",
    "audio_classifier_scaler.pkl", 
    "audio_classifier_label_encoder.pkl",
    "audio_classifier_metadata.pkl"
]

for filename in files:
    print(f"\n--- Testing {filename} ---")
    try:
        # Cek apakah file ada
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue
            
        # Cek ukuran file
        size = os.path.getsize(filename)
        print(f"✓ File exists, size: {size} bytes")
        
        # Coba baca file
        with open(filename, 'rb') as f:
            # Baca beberapa bytes pertama
            first_bytes = f.read(20)
            print(f"First bytes (hex): {first_bytes.hex()}")
            
            # Reset ke awal file
            f.seek(0)
            
            # Coba load pickle
            data = pickle.load(f)
            print(f"✓ Successfully loaded: {type(data)}")
            
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")

print("\n" + "="*50)
print("Test completed!")
