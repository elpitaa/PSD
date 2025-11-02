"""
Script untuk mencoba recovery file pickle yang corrupt
"""
import pickle
import os

def try_load_with_joblib():
    """Coba load dengan joblib sebagai alternatif"""
    try:
        import joblib
        print("Trying with joblib...")
        model = joblib.load('audio_classifier.pkl')
        print("✓ Success with joblib!")
        return model
    except Exception as e:
        print(f"✗ Joblib failed: {e}")
        return None

def try_load_with_different_protocols():
    """Coba load dengan berbagai pickle protocol"""
    for protocol in range(6):
        try:
            print(f"Trying pickle protocol {protocol}...")
            with open('audio_classifier.pkl', 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            print(f"✓ Success with protocol {protocol}!")
            return model
        except Exception as e:
            print(f"✗ Protocol {protocol} failed: {type(e).__name__}")
    return None

def check_file_integrity():
    """Cek integrity file"""
    print("\n=== File Integrity Check ===")
    with open('audio_classifier.pkl', 'rb') as f:
        data = f.read()
        print(f"File size: {len(data)} bytes")
        print(f"First 50 bytes (hex): {data[:50].hex()}")
        print(f"Last 50 bytes (hex): {data[-50:].hex()}")
        
        # Cek apakah ada karakter aneh
        tab_count = data.count(b'\x09')
        print(f"Tab characters (\\x09) found: {tab_count}")
        
        if tab_count > 0:
            print("\n⚠️  Warning: Found tab characters, file might be corrupted!")
            print("Possible causes:")
            print("- File was edited as text")
            print("- File was incorrectly transferred")
            print("- File was saved with wrong encoding")

if __name__ == "__main__":
    os.chdir(r"E:\Semester 5\Proyek Sain Data\project_2")
    
    check_file_integrity()
    
    print("\n=== Recovery Attempts ===")
    
    # Method 1: Joblib
    model = try_load_with_joblib()
    
    if model is None:
        # Method 2: Different protocols
        model = try_load_with_different_protocols()
    
    if model is None:
        print("\n❌ All recovery methods failed!")
        print("\n💡 Recommendations:")
        print("1. Re-generate the model from your training script/notebook")
        print("2. Check if you have a backup of the model file")
        print("3. Make sure the file wasn't opened/edited as text")
        print("4. If copying from another location, use binary transfer mode")
    else:
        print("\n✓ Model recovered successfully!")
        print(f"Model type: {type(model)}")
        
        # Save recovered model
        try:
            with open('audio_classifier_recovered.pkl', 'wb') as f:
                pickle.dump(model, f)
            print("✓ Recovered model saved as 'audio_classifier_recovered.pkl'")
        except Exception as e:
            print(f"✗ Could not save recovered model: {e}")
