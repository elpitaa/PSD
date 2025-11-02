import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import os

# ===============================
# Load Model & Scaler
# ===============================
# Pastikan path sesuai dengan folder app.py
base_path = os.path.dirname(__file__)
model = load(os.path.join(base_path, "knn_model.pkl"))
scaler = load(os.path.join(base_path, "minmax_scaler.pkl"))

# Tentukan berapa fitur yang diharapkan
expected_n_features = None
if hasattr(scaler, "n_features_in_"):
    expected_n_features = int(scaler.n_features_in_)
elif hasattr(model, "n_features_in_"):
    expected_n_features = int(model.n_features_in_)

# ===============================
# Judul Aplikasi
# ===============================
st.title("Prediksi Nilai NO₂ Satu Hari Kedepan dengan KNN Regression")
st.markdown(
    """
    Aplikasi ini memprediksi konsentrasi **NO₂ (Nitrogen Dioksida)** satu hari ke depan
    berdasarkan data input fitur lingkungan.
    
    Model yang digunakan: **K-Nearest Neighbors (KNN) Regression**
    """
)

# ===============================
# Input Fitur
# ===============================
st.subheader("Masukkan Nilai Fitur")

default_feature_names = [
    "PM10", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed"
]

if expected_n_features is None:
    feature_names = default_feature_names
else:
    if expected_n_features == len(default_feature_names):
        feature_names = default_feature_names
    else:
        feature_names = [f"Feature_{i+1}" for i in range(expected_n_features)]
        st.warning(
            f"Model saat ini mengharapkan {expected_n_features} fitur.\n"
            "Gunakan urutan fitur yang sama saat model dilatih.\n"
            "Jika Anda ingin memakai nama fitur yang spesifik, edit kode aplikasi ini."
        )
        try:
            st.caption(f"Info scaler: n_features_in_={getattr(scaler,'n_features_in_',None)}, "
                       f"min_={getattr(scaler,'min_',None)}, scale_={getattr(scaler,'scale_',None)}")
        except Exception:
            pass

# ===============================
# Default Input (nilai pertengahan)
# ===============================
default_inputs = None
try:
    if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
        mid_scaled = np.full(int(expected_n_features or len(feature_names)), 0.5)
        default_inputs = ((mid_scaled - scaler.min_) / scaler.scale_).astype(float)
except Exception:
    default_inputs = None

inputs = []
col1, col2 = st.columns(2)
for i, name in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        default_val = float(default_inputs[i]) if (default_inputs is not None and i < len(default_inputs)) else 0.0
        value = st.number_input(f"{name}", value=default_val)
        inputs.append(value)

# ===============================
# Prediksi
# ===============================
if st.button("Prediksi NO₂"):
    data = np.array(inputs).reshape(1, -1)
    try:
        # Normalisasi input dengan scaler
        data_scaled = scaler.transform(data)
        # Prediksi dengan model
        pred = model.predict(data_scaled)
        pred_val = float(pred[0])

        # ==========================
        # 💡 Denormalisasi hasil prediksi
        # Ganti nilai di bawah ini sesuai rentang data asli NO₂ kamu saat training
        y_min, y_max = 5.0, 200.0
        pred_real = pred_val * (y_max - y_min) + y_min
        # ==========================

        # Tampilkan hasil
        st.success(f"Prediksi Konsentrasi NO₂ Besok: {pred_real:.2f} µg/m³")
        st.caption(f"(Nilai mentah model: {pred_val:.6f})")

        with st.expander("Detail input & transform"):
            st.write({
                'raw_input': data.tolist(),
                'scaled_input': data_scaled.tolist(),
                'prediction_raw': pred.tolist(),
                'prediction_real': pred_real
            })
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")

# ===============================
# Catatan Tambahan
# ===============================
st.markdown("---")
st.caption("Dibuat menggunakan Streamlit | Model: KNN Regression")
