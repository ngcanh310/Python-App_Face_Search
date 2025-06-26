from app.face_detector import extract_face
from app.embedder import get_embedding
import numpy as np

def search(image_path, index, image_paths, top_k=3):
    # Hàm này tìm kiếm các khuôn mặt giống nhất trong chỉ mục FAISS từ một ảnh đầu vào

    # Bước 1: Trích xuất khuôn mặt từ ảnh đầu vào
    face = extract_face(image_path)
    
    # Nếu không phát hiện được khuôn mặt, trả về danh sách rỗng
    if face is None:
        return []

    # Bước 2: Tính vector đặc trưng (embedding) cho khuôn mặt vừa trích xuất
    # Chuyển về float32 và reshape để phù hợp với đầu vào của FAISS
    query_vector = get_embedding(face).astype('float32').reshape(1, -1)

    # Bước 3: Tìm top_k vector gần nhất trong chỉ mục FAISS
    distances, indices = index.search(query_vector, top_k)

    # Bước 4: Trả về danh sách các ảnh gần nhất cùng với khoảng cách tương ứng
    results = [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]

    return results
