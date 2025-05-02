import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gdown
import os

# ตรวจสอบและดาวน์โหลดโมเดล (จาก Google Drive)
if not os.path.exists("best_model.pt"):
    url = "https://drive.google.com/uc?id=10m3Hhi3dNKr7lsFVbL_MpkDmqise9G6u"  # แก้ลิงก์ให้ใช้ได้กับ gdown
    try:
        gdown.download(url, "best_model.pt", quiet=False)
    except Exception as e:
        st.error(f"ดาวน์โหลดโมเดลล้มเหลว: {e}")
        st.stop()

# โหลดโมเดล YOLOv8 (ใช้ cache เพื่อไม่โหลดซ้ำ)
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# ส่วนติดต่อผู้ใช้
st.title("🔍 Real-Time Object Detection By Siwayu Seeyangnok")
st.markdown("อัปโหลดภาพเพื่อให้ระบบตรวจจับวัตถุในภาพด้วยโมเดล YOLOv8 ที่ฝึกมา")

# Slider สำหรับกำหนด Confidence Threshold
conf_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# อัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # โหลดภาพ
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # ทำนายด้วยโมเดล
        with st.spinner("🧠 กำลังประมวลผล..."):
            results = model.predict(image, conf=conf_threshold)

        # แปลงภาพผลลัพธ์ (BGR → RGB)
        result_np = results[0].plot()
        result_rgb = Image.fromarray(result_np[..., ::-1])
        st.image(result_rgb, caption="✅ Prediction Result", use_container_width=True)

        # แสดงผลลัพธ์แบบละเอียด
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.warning("🚫 ไม่พบวัตถุในภาพ! ลองใช้ภาพอื่น หรือปรับ Confidence Threshold.")
        else:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy().round(2)
            class_names_all = model.names  # ใช้จากตัวโมเดลโดยตรง
            class_names = [class_names_all[i] for i in class_ids]

            st.subheader("📋 Detection Details")
            for i, (name, conf) in enumerate(zip(class_names, confidences)):
                st.write(f"{i+1}. **{name}** ({conf*100:.1f}%)")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}") 
