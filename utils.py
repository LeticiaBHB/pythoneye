import cv2

def draw_landmarks(frame, landmarks, indices, color=(0, 0, 255), radius=2):
    """
    Desenha círculos nos pontos especificados dos landmarks.
    """
    h, w, _ = frame.shape
    for idx in indices:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), radius, color, -1)

def euclidean_distance(p1, p2):
    """
    Calcula a distância euclidiana entre dois pontos de landmark.
    """
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5
