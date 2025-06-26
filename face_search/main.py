import os
import glob
import numpy as np
from PIL import Image

from app.face_detector import extract_face         # Hàm cắt khuôn mặt từ ảnh
from app.embedder import get_embedding             # Hàm tạo vector đặc trưng từ khuôn mặt
from app.indexer import create_index               # Hàm tạo FAISS index từ các vector đặc trưng

# --- Đường dẫn ---
IMAGE_DIR = "data/images/"                         # Thư mục chứa ảnh gốc
FACE_DIR = "data/faces/"                           # Thư mục lưu khuôn mặt đã cắt
EMBEDDING_FILE = "data/embeddings.npy"             # File lưu mảng embeddings
IMAGE_PATHS_FILE = "data/image_paths.txt"          # File lưu đường dẫn của khuôn mặt đã cắt
FAISS_INDEX_FILE = "data/faiss.index"              # File lưu FAISS index

# --- Tạo thư mục nếu chưa có ---
os.makedirs(FACE_DIR, exist_ok=True)               # Đảm bảo thư mục lưu khuôn mặt tồn tại

# --- Xử lý tất cả ảnh ---
embeddings = []                                    # Danh sách lưu các embedding
image_paths = []                                   # Danh sách lưu đường dẫn tương ứng

# Lấy danh sách tất cả file ảnh trong IMAGE_DIR
image_files = glob.glob(os.path.join(IMAGE_DIR, "*"))
print(f"Tổng số ảnh tìm thấy: {len(image_files)}")

for img_path in image_files:
    try:
        # Bước 1: Trích xuất khuôn mặt từ ảnh
        face = extract_face(img_path)
        print(f"Checksum face {img_path}: {np.sum(face)}")
        print(f"Face checksum: {np.sum(face)} | from {os.path.basename(img_path)}")
        
        # Nếu không có khuôn mặt hoặc không hợp lệ thì bỏ qua
        if face is None or not isinstance(face, np.ndarray):
            print(f"Không tìm thấy mặt trong: {img_path}")
            continue

        # Bước 2: Trích xuất embedding từ khuôn mặt
        emb = get_embedding(face)
        print(f"Embedding: {emb[:5]}...")

        if emb is None or not isinstance(emb, np.ndarray):
            print(f"Không thể trích xuất đặc trưng từ {img_path}")
            continue

        # Bước 3: Lưu ảnh khuôn mặt đã cắt vào thư mục FACE_DIR
        name = os.path.basename(img_path)
        face_save_path = os.path.join(FACE_DIR, name)
        Image.fromarray(face).save(face_save_path)

        # Bước 4: Lưu embedding và đường dẫn ảnh tương ứng
        embeddings.append(emb)
        image_paths.append(face_save_path)
        print(f"Đã xử lý: {name}")

    except Exception as e:
        # Ghi log nếu gặp lỗi bất kỳ khi xử lý ảnh
        print(f"Lỗi xử lý ảnh {img_path}: {e}")

# --- Lưu dữ liệu ---
if embeddings:
    embeddings = np.array(embeddings, dtype=np.float32)     # Chuyển danh sách về mảng numpy
    np.save(EMBEDDING_FILE, embeddings)                      # Lưu embeddings ra file .npy

    # Ghi đường dẫn ảnh khuôn mặt vào file text
    with open(IMAGE_PATHS_FILE, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"\nĐã lưu {len(embeddings)} vector đặc trưng vào {EMBEDDING_FILE}")
    print(f"Đã lưu danh sách ảnh vào {IMAGE_PATHS_FILE}")

    # --- Tạo và lưu FAISS index ---
    import faiss
    index = create_index(embeddings)                         # Tạo FAISS index từ embeddings
    faiss.write_index(index, FAISS_INDEX_FILE)               # Lưu index ra file
    print(f"Đã lưu FAISS index vào {FAISS_INDEX_FILE}")

else:
    print("Không có ảnh nào được xử lý thành công.")

# --- Thống kê kết quả ---
print(f"\nTổng ảnh đầu vào: {len(image_files)}")
print(f"Số ảnh xử lý thành công: {len(embeddings)}")
print(f"Số ảnh lỗi hoặc không có mặt: {len(image_files) - len(embeddings)}")
