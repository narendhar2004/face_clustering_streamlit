# 👥 Face Clustering & Group Image Matcher

A Streamlit app to find and download group images containing similar faces using face encodings and cosine similarity.

---

## 🔍 What It Does

- Upload a reference image
- Detect and encode all faces from unlabeled group photos
- Compare the reference face to stored encodings using cosine similarity
- Display and download matched group images as a ZIP file

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/narendhar2004/face_clustering_streamlit.git
cd face_clustering_streamlit
pip install -r requirements.txt
streamlit run app.py
```
## 📁 Project Structure

```bash
face-clustering-streamlit/
├── group_images/                             
├── app.py                        
├── face_encodings_grouped.json  
├── requirements.txt             
└── README.md    
```
