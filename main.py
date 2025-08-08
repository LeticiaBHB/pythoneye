import cv2
import mediapipe as mp
import numpy as np
import time

class Calibration:
    def __init__(self, nb_frames=20):
        self.nb_frames = nb_frames
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def get_threshold(self, side):
        if side == 0 and self.thresholds_left:
            return int(np.mean(self.thresholds_left))
        elif side == 1 and self.thresholds_right:
            return int(np.mean(self.thresholds_right))
        else:
            return 50  # valor default razoável

    @staticmethod
    def iris_size(bin_img):
        crop = bin_img[5:-5, 5:-5]
        total_pixels = crop.size
        black_pixels = total_pixels - cv2.countNonZero(crop)
        return black_pixels / total_pixels

    @staticmethod
    def binarize_eye(eye_frame, threshold):
        gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return bin_img

    def find_best_threshold(self, eye_frame):
        average_iris_size = 0.48
        candidates = {}
        for t in range(5, 100, 5):
            bin_img = self.binarize_eye(eye_frame, t)
            size = self.iris_size(bin_img)
            candidates[t] = abs(size - average_iris_size)
        best_thresh = min(candidates, key=candidates.get)
        return best_thresh

    def evaluate(self, eye_frame, side):
        best_thresh = self.find_best_threshold(eye_frame)
        if side == 0:
            self.thresholds_left.append(best_thresh)
            if len(self.thresholds_left) > self.nb_frames:
                self.thresholds_left.pop(0)
        elif side == 1:
            self.thresholds_right.append(best_thresh)
            if len(self.thresholds_right) > self.nb_frames:
                self.thresholds_right.pop(0)

# Configuração MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

WIDTH, HEIGHT = 1366, 768
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

ball_x, ball_y = WIDTH // 2, HEIGHT // 2
smoothing_factor = 0.15
active_eye = 0

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_BORDER = [33, 133, 159, 145, 153, 154]
RIGHT_EYE_BORDER = [362, 263, 386, 374, 380, 381]

LEFT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 2

blink_counter = 0
blink_detected = False
red_start_time = 0
red_duration = 10

sensitivity = 1.3

calibration = Calibration(nb_frames=20)

def draw_button(frame, text, position, is_active):
    x, y = position
    color_bg = (0, 200, 0) if is_active else (60, 60, 60)
    color_border = (255, 255, 255)
    cv2.rectangle(frame, (x, y), (x + 200, y + 50), color_bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 200, y + 50), color_border, 2, cv2.LINE_AA)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = x + (200 - text_size[0]) // 2
    text_y = y + (50 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def get_eye_direction(norm_x, norm_y):
    center_x, center_y = 0.5, 0.5
    dx = norm_x - center_x
    dy = center_y - norm_y
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    if 337.5 <= angle or angle < 22.5:
        return "Leste"
    elif 22.5 <= angle < 67.5:
        return "Nordeste"
    elif 67.5 <= angle < 112.5:
        return "Norte"
    elif 112.5 <= angle < 157.5:
        return "Noroeste"
    elif 157.5 <= angle < 202.5:
        return "Oeste"
    elif 202.5 <= angle < 247.5:
        return "Sudoeste"
    elif 247.5 <= angle < 292.5:
        return "Sul"
    elif 292.5 <= angle < 337.5:
        return "Sudeste"

def eye_aspect_ratio(landmarks, eye_points, frame_w, frame_h):
    def euclid_dist(p1, p2):
        return np.linalg.norm(np.array([p1.x * frame_w, p1.y * frame_h]) - np.array([p2.x * frame_w, p2.y * frame_h]))
    p = [landmarks[i] for i in eye_points]
    A = euclid_dist(p[1], p[5])
    B = euclid_dist(p[2], p[4])
    C = euclid_dist(p[0], p[3])
    ear = (A + B) / (2.0 * C)
    return ear

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    draw_button(frame, "Olho Esquerdo", (WIDTH // 2 - 220, 20), active_eye == 0)
    draw_button(frame, "Olho Direito", (WIDTH // 2 + 20, 20), active_eye == 1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1'):
        active_eye = 0
    elif key == ord('2'):
        active_eye = 1

    direction = "Desconhecida"
    ear = 0.0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        if active_eye == 0:
            iris_landmarks = [face_landmarks.landmark[i] for i in LEFT_IRIS]
            eye_border = [face_landmarks.landmark[i] for i in LEFT_EYE_BORDER]
            ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_EAR_POINTS, w, h)
        else:
            iris_landmarks = [face_landmarks.landmark[i] for i in RIGHT_IRIS]
            eye_border = [face_landmarks.landmark[i] for i in RIGHT_EYE_BORDER]
            ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_EAR_POINTS, w, h)

        iris_x = np.mean([p.x for p in iris_landmarks])
        iris_y = np.mean([p.y for p in iris_landmarks])

        eye_min_x = min([p.x for p in eye_border])
        eye_max_x = max([p.x for p in eye_border])
        eye_min_y = min([p.y for p in eye_border])
        eye_max_y = max([p.y for p in eye_border])

        # Normaliza posição íris no olho
        norm_x = (iris_x - eye_min_x) / (eye_max_x - eye_min_x)
        norm_y = (iris_y - eye_min_y) / (eye_max_y - eye_min_y)
        norm_y = 1 - norm_y

        # Recorta região do olho para calibração
        # Converte landmarks para pixels
        min_x_px = int(eye_min_x * w)
        max_x_px = int(eye_max_x * w)
        min_y_px = int(eye_min_y * h)
        max_y_px = int(eye_max_y * h)

        # Evita erros se região inválida
        if max_x_px - min_x_px > 10 and max_y_px - min_y_px > 10:
            eye_crop = frame[min_y_px:max_y_px, min_x_px:max_x_px]
            # Avalia calibração no olho ativo
            calibration.evaluate(eye_crop, active_eye)

        direction = get_eye_direction(norm_x, norm_y)

        # Aplica sensibilidade e suavização
        target_x = int(norm_x * WIDTH * sensitivity)
        target_y = int(norm_y * HEIGHT * sensitivity)
        target_x = np.clip(target_x, 0, WIDTH)
        target_y = np.clip(target_y, 0, HEIGHT)

        ball_x = int(ball_x * (1 - smoothing_factor) + target_x * smoothing_factor)
        ball_y = int(ball_y * (1 - smoothing_factor) + target_y * smoothing_factor)

        # Detecta piscar via EAR
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                blink_detected = True
                red_start_time = time.time()
            blink_counter = 0

    # Bola com cor muda ao piscar
    if blink_detected:
        elapsed = time.time() - red_start_time
        if elapsed < red_duration:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
            blink_detected = False
    else:
        color = (255, 0, 0)

    cv2.circle(frame, (ball_x, ball_y), 20, color, -1)
    cv2.putText(frame, f"Olho ativo: {'Esquerdo' if active_eye == 0 else 'Direito'}", (30, HEIGHT - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if results.multi_face_landmarks:
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Exibe direção
    cv2.putText(frame, f"Direcao: {direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Indicação de calibração
    if calibration.is_complete():
        thresh = calibration.get_threshold(active_eye)
        cv2.putText(frame, f"Threshold calibrado: {thresh}", (30, HEIGHT - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Calibrando...", (30, HEIGHT - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Pressione 1 para Olho Esquerdo | 2 para Olho Direito | ESC para sair", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Controle com os Olhos", frame)

cap.release()
cv2.destroyAllWindows()
