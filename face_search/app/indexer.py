import faiss
import numpy as np

def create_index(embeddings):
    # Hàm này tạo một chỉ mục (index) FAISS để tìm kiếm vector gần nhất (nearest neighbors)
    # embeddings: numpy array dạng (số mẫu, số chiều), ví dụ (1000, 512)

    d = embeddings.shape[1]  # Số chiều của mỗi vector (embedding)

    # Tạo index dùng khoảng cách L2 (Euclidean distance) để so sánh các vector
    index = faiss.IndexFlatL2(d)

    # Thêm toàn bộ embedding vào index
    index.add(embeddings)

    # Trả về index đã khởi tạo
    return index
