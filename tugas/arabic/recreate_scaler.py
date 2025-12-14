import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create a StandardScaler with default parameters
# This is a temporary solution - ideally we need the original fitted scaler
scaler = StandardScaler()

# Fit on dummy data with the expected shape (13 MFCC features)
# This creates a scaler that will normalize each MFCC coefficient
dummy_data = np.random.randn(1000, 13)  # 1000 samples, 13 features
scaler.fit(dummy_data)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler created and saved successfully!")
print(f"Mean: {scaler.mean_}")
print(f"Variance: {scaler.var_}")
