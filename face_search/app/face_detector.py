from PIL import Image
import numpy as np
from mtcnn import MTCNN

# Khởi tạo detector khuôn mặt từ thư viện MTCNN
detector = MTCNN()

def extract_face(image_path, required_size=(160, 160), resize_input=True):
    # Hàm này dùng để cắt khuôn mặt từ ảnh đầu vào và resize về kích thước chuẩn (160x160)
    
    # Mở ảnh từ đường dẫn và chuyển sang RGB
    img = Image.open(image_path).convert('RGB')

    # Nếu ảnh quá lớn, thu nhỏ lại để tăng tốc độ xử lý
    if resize_input:
        max_size = 800
        if max(img.size) > max_size:
            # Giữ nguyên tỉ lệ khi thu nhỏ ảnh, dùng thuật toán nội suy LANCZOS cho chất lượng cao
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Chuyển ảnh PIL sang numpy array
    pixels = np.asarray(img)

    # Dò khuôn mặt trong ảnh
    results = detector.detect_faces(pixels)

    # Nếu không tìm thấy khuôn mặt nào, trả về None
    if len(results) == 0:
        return None

    # Lấy toạ độ hộp chứa khuôn mặt đầu tiên
    x1, y1, width, height = results[0]['box']
    
    # Đảm bảo toạ độ không bị âm
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + width, y1 + height

    # Cắt khuôn mặt từ ảnh gốc
    face = pixels[y1:y2, x1:x2]

    # Resize khuôn mặt về kích thước yêu cầu (mặc định là 160x160)
    face_image = Image.fromarray(face).resize(required_size)

    # Trả về khuôn mặt dưới dạng numpy array
    return np.asarray(face_image)
