import os
import json
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity

# Settings
input_dir = "group_images"
json_path = "face_encodings_grouped.json"
similarity_threshold = 0.90

face_dict = {}

def encoding_to_key(enc, decimals=5):
    # Round and stringify the encoding to use as key
    return ",".join([str(round(val, decimals)) for val in enc])

# Process images
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    full_path = os.path.join(input_dir, filename)
    image = face_recognition.load_image_file(full_path)
    encodings = face_recognition.face_encodings(image)

    for new_enc in encodings:
        new_key = encoding_to_key(new_enc)

        # Check if it matches any existing encoding
        matched = False
        for existing_key in list(face_dict.keys()):
            existing_enc = np.array([float(x) for x in existing_key.split(",")])
            similarity = cosine_similarity([existing_enc], [new_enc])[0][0]

            if similarity >= similarity_threshold:
                # Add this image to that existing key
                if full_path not in face_dict[existing_key]:
                    face_dict[existing_key].append(full_path)
                matched = True
                break

        # If no match, add as new key
        if not matched:
            face_dict[new_key] = [full_path]

# Save to JSON
with open(json_path, "w") as f:
    json.dump(face_dict, f, indent=2)

print(f"Saved grouped encodings in: {json_path}")
