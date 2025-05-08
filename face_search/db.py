import numpy as np
import mysql.connector

# --- Cấu hình kết nối MySQL ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""  # 🔒 Thay bằng mật khẩu thật
DB_NAME = "face_search"

# --- Đường dẫn dữ liệu ---
EMBEDDING_FILE = "./data/embeddings.npy"
IMAGE_PATHS_FILE = "./data/image_paths.txt"

# --- Kết nối MySQL ---
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()

# --- Tạo database và bảng nếu chưa có ---
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
cursor.execute(f"USE {DB_NAME}")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INT AUTO_INCREMENT PRIMARY KEY,
        file_path VARCHAR(255) NOT NULL,
        embedding LONGBLOB NOT NULL,
        age INT DEFAULT NULL,
        note TEXT
    )
""")

# --- Đọc dữ liệu từ file ---
print("📥 Đang tải embeddings và đường dẫn ảnh...")
embeddings = np.load(EMBEDDING_FILE)
with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]

# --- Ghi vào MySQL ---
print(f"💾 Đang lưu {len(embeddings)} embeddings vào MySQL...")
for emb, path in zip(embeddings, image_paths):
    emb_bytes = emb.astype(np.float32).tobytes()
    cursor.execute("INSERT INTO faces (file_path, embedding) VALUES (%s, %s)", (path, emb_bytes))

conn.commit()
cursor.close()
conn.close()
print("✅ Đã lưu xong dữ liệu vào MySQL.")
