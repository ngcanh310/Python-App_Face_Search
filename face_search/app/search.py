from app.face_detector import extract_face
from app.embedder import get_embedding
import numpy as np

def search(image_path, index, image_paths, top_k=3):
    face = extract_face(image_path)
    if face is None:
        return []
    query_vector = get_embedding(face).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results
