import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Hybrid Emotion AI")

# --- 1. Load Model Hybrid ---
@st.cache_resource
def load_models():
    try:
        # Load Feature Extractor (CNN)
        cnn_model = tf.keras.models.load_model('emotion_feature_extractor.keras')
        
        # Load Classifier (Random Forest)
        rf_model = joblib.load('emotion_rf_model.pkl')
        
        # Load Labels
        with open('emotion_labels.json', 'r') as f:
            label_map = json.load(f)
        # Pastikan label urut sesuai index (0, 1, 2...)
        labels = [label_map[str(i)] for i in range(len(label_map))]
        
        return cnn_model, rf_model, labels
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

cnn_model, rf_model, class_labels = load_models()

# Load Face Detector
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

st.title("Deteksi Emosi Hybrid (CNN + Random Forest)")

# --- 2. Logika Pemrosesan (Sesuai Notebook pcd2.ipynb) ---
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.cnn_model = cnn_model
        self.rf_model = rf_model
        self.labels = class_labels
        self.face_cascade = face_cascade

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Deteksi Wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop Wajah
            roi_gray = gray[y:y+h, x:x+w]
            
            # --- PREPROCESSING (Wajib sama dengan Notebook) ---
            # 1. Resize ke 96x96 (sesuai pcd2.ipynb)
            roi = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
            
            # 2. Normalisasi & Reshape
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)   # (1, 96, 96)
            roi = np.expand_dims(roi, axis=-1)  # (1, 96, 96, 1)

            try:
                # --- PREDIKSI HYBRID ---
                # Langkah 1: Ekstrak Fitur pakai CNN
                features = self.cnn_model.predict(roi, verbose=0)
                
                # Langkah 2: Prediksi pakai Random Forest
                prediction_idx = self.rf_model.predict(features)[0]
                proba = self.rf_model.predict_proba(features)[0]
                confidence = np.max(proba) * 100
                
                label_text = f"{self.labels[prediction_idx]} ({confidence:.1f}%)"
                
                # Gambar Kotak
                color = (0, 255, 0) if self.labels[prediction_idx] == 'happy' else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception as e:
                print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. Tampilan WebRTC ---
if cnn_model is not None and rf_model is not None:
    webrtc_streamer(
        key="hybrid-emotion",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=EmotionVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    st.warning("Model belum dimuat dengan benar. Cek file .keras, .pkl, dan .json Anda.")
