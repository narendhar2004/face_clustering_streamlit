import streamlit as st
import face_recognition
import numpy as np
import json
from PIL import Image
import os
import io
import zipfile

# Configuration
JSON_PATH = "face_clusters_dbscan.json"
DISTANCE_THRESHOLD = 0.6   # Euclidean distance threshold

st.title("👥 Face Clustering Viewer (Euclidean Distance - DBSCAN)")
st.subheader("Upload a reference image to find matching group images")

# Load JSON
@st.cache_data
def load_encodings():
    with open(JSON_PATH, "r") as f:
        return json.load(f)

clusters = load_encodings()

# File uploader
uploaded_file = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Reference Image", use_column_width=True)

    # Encode face
    image_np = np.array(image)
    ref_encodings = face_recognition.face_encodings(image_np)

    if not ref_encodings:
        st.error("❌ No face found in the uploaded image.")
    else:
        ref_enc = ref_encodings[0]

        st.info(f"Finding matches with distance ≤ {DISTANCE_THRESHOLD}")
        matched = []
        matched_image_paths = set()

        # Loop through clusters
        for cluster in clusters:
            for stored_enc, img_path in zip(cluster["encodings"], cluster["images"]):
                stored_enc = np.array(stored_enc)

                # Euclidean distance
                distance = np.linalg.norm(stored_enc - ref_enc)

                if distance <= DISTANCE_THRESHOLD:
                    matched.append((cluster["id"], distance, img_path))
                    matched_image_paths.add(img_path)

        if matched:
            # Sort by distance ascending
            matched.sort(key=lambda x: x[1])

            for idx, (cluster_id, dist, img_path) in enumerate(matched):
                with st.expander(f"Cluster {cluster_id} - Match #{idx+1} (Distance: {dist:.4f})"):
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path), width=250)
                    else:
                        st.warning(f"Image not found: {img_path}")

            # ✅ ZIP download
            if matched_image_paths:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for img_path in matched_image_paths:
                        if os.path.exists(img_path):
                            zip_file.write(img_path, arcname=os.path.basename(img_path))

                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download All Matched Images as ZIP",
                    data=zip_buffer,
                    file_name="matched_images.zip",
                    mime="application/zip"
                )
        else:
            st.warning("No matching faces found.")
