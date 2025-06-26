import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import faiss
import mysql.connector

from app.face_detector import extract_face       # Hàm cắt khuôn mặt
from app.embedder import get_embedding           # Hàm tạo embedding từ khuôn mặt

# --- Kết nối MySQL và lấy dữ liệu ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "face_search"

# Kết nối database và truy vấn toàn bộ bảng 'faces'
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)
cursor = conn.cursor()
cursor.execute("SELECT file_path, embedding FROM faces")
rows = cursor.fetchall()
conn.close()

# Tách dữ liệu thành image_paths và embeddings
image_paths = []
embeddings = []
for path, emb_blob in rows:
    image_paths.append(path)                              # Lưu đường dẫn ảnh
    emb_array = np.frombuffer(emb_blob, dtype=np.float32) # Giải mã blob về mảng numpy
    embeddings.append(emb_array)
embeddings = np.array(embeddings).astype('float32')       # Chuyển sang mảng float32 để dùng với FAISS

# --- FAISS index ---
index = faiss.IndexFlatL2(embeddings.shape[1])            # Tạo index sử dụng khoảng cách L2
index.add(embeddings)                                     # Thêm embeddings vào index

# --- Tạo GUI ---
root = tk.Tk()
root.title("Tìm kiếm khuôn mặt tương tự")

frame = tk.Frame(root)
frame.pack(pady=10)

# --- Khu vực hiển thị kết quả ---
results_frame = tk.Frame(root)
results_frame.pack(pady=10)

def clear_results():
    # Xóa toàn bộ widget cũ trong khu vực hiển thị kết quả
    for widget in results_frame.winfo_children():
        widget.destroy()

def select_image():
    # Mở hộp thoại chọn ảnh
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    clear_results()  # Xóa kết quả cũ

    # Trích xuất khuôn mặt từ ảnh đã chọn
    face = extract_face(file_path)
    if face is None:
        tk.Label(results_frame, text="Không phát hiện được khuôn mặt.").pack()
        return

    # --- Hiển thị ảnh gốc đã chọn ---
    original_img = Image.open(file_path).resize((150, 150))
    original_tk = ImageTk.PhotoImage(original_img)
    label = tk.Label(results_frame, image=original_tk, text="Ảnh đã chọn", compound="top", font=("Arial", 10, "bold"))
    label.image = original_tk  # Giữ tham chiếu để không bị thu hồi bộ nhớ
    label.pack(pady=5)

    # --- Dòng tiêu đề ---
    tk.Label(results_frame, text="Kết quả lần lượt 3 ảnh giống nhất:", font=("Arial", 12, "bold")).pack(pady=(10, 10))

    # --- Tìm kiếm ảnh giống ---
    query_embedding = get_embedding(face).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding, 4)  # Lấy 3 kết quả gần nhất (lọc sau nếu trùng)

    shown = 0
    result_container = tk.Frame(results_frame)
    result_container.pack()

    for rank, idx in enumerate(I[0]):
        result_path = image_paths[idx]

        # Bỏ qua nếu ảnh kết quả chính là ảnh đầu vào
        if os.path.basename(result_path) == os.path.basename(file_path):
            continue

        img = Image.open(result_path).resize((150, 150))
        tk_img = ImageTk.PhotoImage(img)

        label = tk.Label(result_container, image=tk_img, text=f"Top {shown+1}", compound="top", padx=10)
        label.image = tk_img  # Giữ tham chiếu ảnh
        label.pack(side="left", padx=10)
        shown += 1
        if shown == 3:
            break


# --- Nút chọn ảnh ---
btn = tk.Button(frame, text="Chọn ảnh để tìm kiếm", command=select_image, font=("Arial", 12), bg="lightblue")
btn.pack()

# --- Khởi chạy giao diện ---
root.mainloop()
