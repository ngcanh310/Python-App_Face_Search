import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Tránh lỗi OpenMP

import numpy as np
import faiss
from PIL import Image
import mysql.connector

from app.face_detector import extract_face
from app.embedder import get_embedding

# --- Thông tin kết nối MySQL ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""   # 🔒 Thay bằng mật khẩu thật
DB_NAME = "face_search"

# --- Kết nối tới MySQL và lấy dữ liệu ---
print("📥 Đang tải dữ liệu từ MySQL...")
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
cursor = conn.cursor()

cursor.execute("SELECT file_path, embedding FROM faces")
rows = cursor.fetchall()

image_paths = []
embeddings = []
for path, emb_blob in rows:
    image_paths.append(path)
    emb_array = np.frombuffer(emb_blob, dtype=np.float32)
    embeddings.append(emb_array)

cursor.close()
conn.close()

if not embeddings:
    print("❌ Không có dữ liệu embeddings trong MySQL.")
    exit()

embeddings = np.array(embeddings).astype('float32')

# --- Tạo FAISS index ---
print("⚙️ Tạo FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# --- Ảnh mới để so sánh ---
query_img_path = "./data/faces_focus/image (188).jpg"  # 💡 Thay ảnh đầu vào tại đây
print(f"🔍 Đang xử lý ảnh: {query_img_path}")

face = extract_face(query_img_path)
if face is None:
    print("❌ Không phát hiện được khuôn mặt.")
    exit()

query_embedding = get_embedding(face).astype('float32').reshape(1, -1)

# --- Tìm top 5, sau đó loại trùng để còn 3 ảnh khác ---
D, I = index.search(query_embedding, 5)

print("\n🧠 Kết quả tìm kiếm:")
results_shown = 0
for rank, idx in enumerate(I[0]):
    result_path = image_paths[idx]
    distance = D[0][rank]

    if os.path.basename(result_path) == os.path.basename(query_img_path):
        continue  # Bỏ qua nếu trùng ảnh đầu vào

    print(f"{results_shown + 1}. {result_path} (distance: {distance:.4f})")
    img = Image.open(result_path)
    img.show(title=f"Kết quả {results_shown + 1}")
    
    results_shown += 1
    if results_shown == 3:
        break

# --- In ra Embedding ---
print("🔍 Embedding ảnh truy vấn:")
print(query_embedding)

print("🔍 Embedding ảnh gần nhất trong database:")
print(embeddings[I[0][0]])
