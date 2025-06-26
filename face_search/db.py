import numpy as np
import mysql.connector

# --- Cấu hình kết nối MySQL ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""  # 🔒 Thay bằng mật khẩu thật nếu có
DB_NAME = "face_search"

# --- Đường dẫn dữ liệu ---
EMBEDDING_FILE = "./data/embeddings.npy"           # File chứa embeddings đã lưu bằng numpy
IMAGE_PATHS_FILE = "./data/image_paths.txt"        # File chứa đường dẫn tương ứng của ảnh

# --- Kết nối MySQL ---
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()

# # --- Tạo database và bảng nếu chưa có ---
# # Các dòng dưới đây bị comment lại; nếu muốn chạy lần đầu thì bỏ comment để tạo DB và bảng
# cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")     # Tạo database nếu chưa tồn tại
# cursor.execute(f"USE {DB_NAME}")                               # Sử dụng database vừa tạo
# cursor.execute("""                                             
#     CREATE TABLE IF NOT EXISTS faces (
#         id INT AUTO_INCREMENT PRIMARY KEY,                     # ID tự động tăng
#         file_path VARCHAR(255) NOT NULL,                       # Đường dẫn ảnh
#         embedding LONGBLOB NOT NULL,                           # Dữ liệu embedding dạng nhị phân
#         age INT DEFAULT NULL,                                  # Tuổi (tùy chọn)
#         note TEXT                                               # Ghi chú thêm (tùy chọn)
#     )
# """)

# --- Đọc dữ liệu từ file ---
print("Đang tải embeddings và đường dẫn ảnh...")
embeddings = np.load(EMBEDDING_FILE)                             # Đọc file .npy chứa embeddings
with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]                  # Đọc đường dẫn ảnh từ file text

# --- Ghi vào MySQL ---
print(f"Đang lưu {len(embeddings)} embeddings vào MySQL...")
for emb, path in zip(embeddings, image_paths):
    emb_bytes = emb.astype(np.float32).tobytes()                # Chuyển embedding sang bytes để lưu vào MySQL
    cursor.execute("INSERT INTO faces (file_path, embedding) VALUES (%s, %s)", (path, emb_bytes))

conn.commit()        # Lưu các thay đổi vào database
cursor.close()       # Đóng cursor
conn.close()         # Đóng kết nối MySQL
print("Đã lưu xong dữ liệu vào MySQL.")
