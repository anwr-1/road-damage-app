import os
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detector", layout="centered")

MODEL_PATH = "best.pt"  # place your best-performing YOLO weights here

@st.cache_resource
def load_model(path):
    return YOLO(path)

st.title("🛣️ Road Damage Detector")
st.write(
    "Upload a road photo and this app detects and localizes damage types "
    "(alligator cracking, block cracking, potholes, cracks, transverse/longitudinal "
    "cracks) using a trained YOLO model."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: `{MODEL_PATH}`.\n\n"
        "Download your best-performing YOLO weights (best.pt) from the training "
        "notebook's output and place it in this app's folder."
    )
    st.stop()

model = load_model(MODEL_PATH)

conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting..."):
        results = model.predict(np.array(image), conf=conf_threshold, verbose=False)

    result = results[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    st.image(annotated, caption="Detections", use_container_width=True)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Detected damage")
        rows = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            rows.append({"Class": cls_name, "Confidence": f"{conf:.1%}"})
        st.table(rows)
    else:
        st.info("No damage detected above the confidence threshold.")
else:
    st.info("Upload an image to get started.")
