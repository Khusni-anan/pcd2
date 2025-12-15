import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Emotion AI Lite")

# --- 1. Load Model TFLite (Super Ringan) ---
@st.cache_resource
def load_model():
    try:
        # Load TFLite
        interpreter = tf.lite.Interpreter(model_path="final_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

interpreter = load_model()

# Setup Tensor Index
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Load Face Detector
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

# Label Emosi (Urutkan sesuai folder dataset Anda, biasanya urutan alfabet)
# Cek di Colab: print(list(emotion_map.values())) untuk memastikan urutannya
LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

st.title("Deteksi Emosi Wajah (TFLite Version) 🚀")

# --- 2. Logika Pemrosesan ---
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = face_cascade
        self.interpreter = interpreter
        
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Deteksi Wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop Wajah
            roi_gray = gray[y:y+h, x:x+w]
            
            # Preprocessing (Wajib sama dengan training: 96x96, normalize)
            roi = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            if self.interpreter:
                # --- PREDIKSI TFLITE ---
                self.interpreter.set_tensor(input_details[0]['index'], roi)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(output_details[0]['index'])
                
                # Ambil hasil tertinggi
                idx = np.argmax(output_data[0])
                confidence = np.max(output_data[0]) * 100
                label_text = f"{LABELS[idx]} ({confidence:.1f}%)"
                
                # Warna-warni
                color = (0, 255, 0) if LABELS[idx] == 'Happy' else (0, 0, 255)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. WebRTC ---
if interpreter:
    webrtc_streamer(
        key="emotion-lite",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=EmotionVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    st.error("Model final_model.tflite tidak ditemukan di GitHub!")
