# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time

# -----------------------------
# CONFIG PAGE
# -----------------------------
st.set_page_config(
    page_title="Custom Object Detection",
    page_icon="🎯",
    layout="wide",
)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
# GANTI model_path ke model custom kamu
# Jika belum ada, biarkan saja pakai YOLO default
MODEL_PATH = "yolov8n.pt"   # misalnya "best.pt" kalau sudah punya model custom

# Load model
model = YOLO(MODEL_PATH)

# -----------------------------
# CENTER LAYOUT
# -----------------------------
left, center, right = st.columns([1, 2, 1])

with center:
    st.title("🎯 Custom Object Detection App")
    st.write("Upload **gambar** atau **video** untuk mendeteksi objek menggunakan model YOLO custom.")

    # Choose input type
    input_type = st.radio(
        "Pilih Jenis Input:",
        ["Gambar", "Video"],
        horizontal=True,
    )
    
    # Upload file
    uploaded_file = st.file_uploader(
        f"Unggah file {input_type.lower()}",
        type=["jpg", "jpeg", "png"] if input_type == "Gambar" else ["mp4", "mov", "avi", "mkv"],
    )
    
    if uploaded_file:
        if input_type == "Gambar":
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_column_width=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image.save(temp_file.name)
                temp_path = temp_file.name

            with st.spinner("🔍 Melakukan deteksi objek..."):
                results = model.predict(temp_path, save=True, imgsz=640, conf=0.5)
                
            result_path = results[0].save_dir + "/" + os.path.basename(temp_path)
            st.success("✅ Deteksi selesai!")
            st.image(result_path, caption="Hasil Deteksi", use_column_width=True)

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            st.video(tfile.name, format="video/mp4")
            
            st.info("🔍 Mulai deteksi video... Ini mungkin memerlukan waktu beberapa detik atau menit tergantung panjang video.")
            
            progress_bar = st.progress(0, text="Sedang memproses...")

            # Simulasi progress (untuk demo)
            for percent_complete in range(0, 100, 5):
                time.sleep(0.1)
                progress_bar.progress(percent_complete + 5, text=f"Proses: {percent_complete+5}%")
            
            with st.spinner("🔍 Melakukan deteksi objek pada video..."):
                results = model.predict(tfile.name, save=True, imgsz=640, conf=0.5)
            
            progress_bar.empty()
            st.success("✅ Video selesai diproses!")
            
            result_dir = results[0].save_dir
            result_files = os.listdir(result_dir)
            video_files = [f for f in result_files if f.endswith(".mp4")]
            
            if video_files:
                result_video_path = os.path.join(result_dir, video_files[0])
                st.video(result_video_path)
            else:
                st.warning("⚠️ Tidak ditemukan video hasil deteksi.")
