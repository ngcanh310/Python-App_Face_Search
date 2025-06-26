import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np

# Khởi tạo thiết bị: sử dụng GPU nếu có, nếu không thì dùng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình InceptionResnetV1 đã được huấn luyện trước trên tập dữ liệu VGGFace2
# Đặt chế độ eval (đánh giá) và chuyển model sang thiết bị tương ứng (CPU/GPU)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face_pixels):
    # Hàm tính embedding (vector đặc trưng) cho khuôn mặt đầu vào

    # Kiểm tra đầu vào có đúng kích thước (160, 160, 3) không
    # Nếu không đúng, raise lỗi
    if face_pixels.shape != (160, 160, 3):
        raise ValueError(f"Kích thước ảnh đầu vào không hợp lệ: {face_pixels.shape}")

    # Chuẩn hóa ảnh: từ [0, 255] -> [0, 1], sau đó -> [-1, 1]
    face_pixels = face_pixels.astype(np.float32) / 255.0
    face_pixels = (face_pixels - 0.5) / 0.5  # Chuẩn hóa về khoảng [-1, 1] như mô hình yêu cầu

    # Chuyển ảnh từ định dạng (H, W, C) sang (1, C, H, W) để phù hợp với đầu vào của mô hình PyTorch
    face_tensor = torch.from_numpy(face_pixels).permute(2, 0, 1).unsqueeze(0).to(device)

    # Tắt chế độ tính toán gradient (tiết kiệm bộ nhớ, tăng tốc)
    with torch.no_grad():
        # Tính embedding bằng model và chuyển về NumPy array
        embedding = model(face_tensor).cpu().numpy()[0]

    # Trả về vector embedding (1D array, thường có kích thước 512 phần tử)
    return embedding
