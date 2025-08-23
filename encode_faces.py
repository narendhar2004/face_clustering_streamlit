import os
import json
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN

# Settings
input_dir = "group_images"
json_path = "face_clusters_dbscan.json"

encodings = []
image_paths = []

# Step 1: Extract all encodings first
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    full_path = os.path.join(input_dir, filename)
    image = face_recognition.load_image_file(full_path)
    faces = face_recognition.face_encodings(image)

    for enc in faces:
        encodings.append(enc)
        image_paths.append(full_path)

encodings = np.array(encodings)

# Step 2: Run DBSCAN clustering
clustering = DBSCAN(
    eps=0.6,        # Distance threshold (tune this!)
    min_samples=2,   # At least 2 faces to form a cluster
    metric="euclidean"
).fit(encodings)

labels = clustering.labels_

# Step 3: Build clusters
clusters = {}
for label, path, enc in zip(labels, image_paths, encodings):
    if label == -1:
        cluster_id = "outlier"
    else:
        cluster_id = int(label)

    if cluster_id not in clusters:
        clusters[cluster_id] = {
            "id": cluster_id,
            "images": [],
            "encodings": []
        }

    clusters[cluster_id]["images"].append(path)
    clusters[cluster_id]["encodings"].append(enc.tolist())

# Step 4: Save to JSON
with open(json_path, "w") as f:
    json.dump(list(clusters.values()), f, indent=2)

print(f"Saved DBSCAN clusters in: {json_path}")
