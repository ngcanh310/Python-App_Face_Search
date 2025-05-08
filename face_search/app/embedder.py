import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np

# Khởi tạo model, tải trọng số pretrained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face_pixels):
    # Resize về (160, 160, 3) nếu cần
    if face_pixels.shape != (160, 160, 3):
        raise ValueError(f"Kích thước ảnh đầu vào không hợp lệ: {face_pixels.shape}")

    # Chuẩn hóa từ [0, 255] -> [0, 1] -> [-1, 1]
    face_pixels = face_pixels.astype(np.float32) / 255.0
    face_pixels = (face_pixels - 0.5) / 0.5  # về [-1, 1]

    # Chuyển sang Tensor (C, H, W)
    face_tensor = torch.from_numpy(face_pixels).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor).cpu().numpy()[0]

    return embedding
