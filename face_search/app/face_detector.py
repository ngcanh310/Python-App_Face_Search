from PIL import Image
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def extract_face(image_path, required_size=(160, 160), resize_input=True):
    img = Image.open(image_path).convert('RGB')

    # Resize ảnh nếu quá to
    if resize_input:
        max_size = 800
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    pixels = np.asarray(img)
    results = detector.detect_faces(pixels)

    if len(results) == 0:
        return None

    x1, y1, width, height = results[0]['box']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face).resize(required_size)
    return np.asarray(face_image)
