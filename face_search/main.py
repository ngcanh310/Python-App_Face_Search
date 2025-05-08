import os
import glob
import numpy as np
from PIL import Image

from app.face_detector import extract_face
from app.embedder import get_embedding
from app.indexer import create_index

# --- Đường dẫn ---
IMAGE_DIR = "data/images/"
FACE_DIR = "data/faces_focus/"
EMBEDDING_FILE = "data/embeddings.npy"
IMAGE_PATHS_FILE = "data/image_paths.txt"
FAISS_INDEX_FILE = "data/faiss.index"

# --- Tạo thư mục nếu chưa có ---
os.makedirs(FACE_DIR, exist_ok=True)

# --- Xử lý tất cả ảnh ---
embeddings = []
image_paths = []

image_files = glob.glob(os.path.join(IMAGE_DIR, "*"))
print(f"🔍 Tổng số ảnh tìm thấy: {len(image_files)}")

for img_path in image_files:
    try:
        face = extract_face(img_path)
        print(f"🧩 Checksum face {img_path}: {np.sum(face)}")
        print(f"🧠 Face checksum: {np.sum(face)} | from {os.path.basename(img_path)}")
        if face is None or not isinstance(face, np.ndarray):
            print(f"⚠️ Không tìm thấy mặt trong: {img_path}")
            continue
        # Trích xuất vector đặc trưng
        emb = get_embedding(face)
        print(f"👉 Embedding: {emb[:5]}...")
        if emb is None or not isinstance(emb, np.ndarray):
            print(f"⚠️ Không thể trích xuất đặc trưng từ {img_path}")
            continue

        # Lưu khuôn mặt đã cắt
        name = os.path.basename(img_path)
        face_save_path = os.path.join(FACE_DIR, name)
        Image.fromarray(face).save(face_save_path)

        # Lưu vector và đường dẫn tương ứng
        embeddings.append(emb)
        image_paths.append(face_save_path)
        print(f"✅ Đã xử lý: {name}")
        
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh {img_path}: {e}")

# --- Lưu dữ liệu ---
if embeddings:
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDING_FILE, embeddings)

    with open(IMAGE_PATHS_FILE, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"\n✅ Đã lưu {len(embeddings)} vector đặc trưng vào {EMBEDDING_FILE}")
    print(f"✅ Đã lưu danh sách ảnh vào {IMAGE_PATHS_FILE}")

    # --- Tạo và lưu FAISS index ---
    import faiss
    index = create_index(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ Đã lưu FAISS index vào {FAISS_INDEX_FILE}")

else:
    print("❌ Không có ảnh nào được xử lý thành công.")
    
print(f"\n🔍 Tổng ảnh đầu vào: {len(image_files)}")
print(f"✅ Số ảnh xử lý thành công: {len(embeddings)}")
print(f"❌ Số ảnh lỗi hoặc không có mặt: {len(image_files) - len(embeddings)}")