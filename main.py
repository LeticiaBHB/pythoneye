import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

WIDTH, HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

LEFT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE_BORDER = [33, 133, 159, 145, 153, 154, 155, 144, 163, 7, 246, 161, 160]
RIGHT_IRIS = [473, 474, 475, 476, 477]
RIGHT_EYE_BORDER = [263, 362, 386, 374, 380, 381, 382, 373, 390, 249, 466, 388, 387]

calib_features = []
calib_targets = []
model_trained = False
model = LinearRegression()

active_eye = 0  # 0 = esquerdo, 1 = direito, 2 = ambos
landmarks_for_calib = None

grid_x = np.linspace(0.1, 0.9, 5)
grid_y = np.linspace(0.1, 0.9, 5)
calib_points = [(x * WIDTH, y * HEIGHT) for y in grid_y for x in grid_x]

calib_idx = 0
frames_per_point = 50
frame_count = 0
temp_features = []

gain = 1.0
GAIN_STEP = 0.05
print("- Pressione 'g' para reduzir o ganho")
print("- Pressione 'h' para aumentar o ganho")

blink_start_time = None
blinking = False
BLINK_DURATION = 5  # segundos
CLICK_THRESHOLD = 5.0  # Threshold ajustado para razão horizontal/vertical

# Filtro de Kalman
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
        self.last_measurement = np.array((2, 1), np.float32)
        self.last_prediction = np.array((2, 1), np.float32)

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        self.last_measurement = measured
        self.last_prediction = predicted
        return predicted[0][0], predicted[1][0]

kf = KalmanFilter()

def extract_eye_features(landmarks, iris_idxs, eye_border_idxs):
    iris_points = [landmarks[i] for i in iris_idxs]
    eye_border_points = [landmarks[i] for i in eye_border_idxs]

    iris_xs = np.array([p.x for p in iris_points])
    iris_ys = np.array([p.y for p in iris_points])
    border_xs = np.array([p.x for p in eye_border_points])
    border_ys = np.array([p.y for p in eye_border_points])

    min_x, max_x = border_xs.min(), border_xs.max()
    min_y, max_y = border_ys.min(), border_ys.max()
    width = max_x - min_x
    height = max_y - min_y

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    iris_mean_x = iris_xs.mean()
    iris_mean_y = iris_ys.mean()

    dist_x = (iris_mean_x - center_x) / width if width > 0 else 0
    dist_y = (iris_mean_y - center_y) / height if height > 0 else 0

    features = [
        (iris_mean_x - min_x) / width if width > 0 else 0,
        (iris_mean_y - min_y) / height if height > 0 else 0,
        dist_x,
        dist_y,
        iris_xs.var(),
        iris_ys.var(),
        np.mean([np.linalg.norm([iris_xs[i] - iris_xs[j], iris_ys[i] - iris_ys[j]])
                 for i in range(len(iris_points)) for j in range(i + 1, len(iris_points))]),
        np.std([np.linalg.norm([iris_xs[i] - iris_xs[j], iris_ys[i] - iris_ys[j]])
                for i in range(len(iris_points)) for j in range(i + 1, len(iris_points))]),
    ]
    return features

def get_blinking_ratio_mediapipe(landmarks, eye_idxs):
    eye_points = [(int(landmarks[i].x * WIDTH), int(landmarks[i].y * HEIGHT)) for i in eye_idxs]
    left_point = np.array(eye_points[0])
    right_point = np.array(eye_points[3])
    center_top = ((eye_points[1][0] + eye_points[2][0]) / 2, (eye_points[1][1] + eye_points[2][1]) / 2)
    center_bottom = ((eye_points[5][0] + eye_points[4][0]) / 2, (eye_points[5][1] + eye_points[4][1]) / 2)
    hor_line_length = np.linalg.norm(left_point - right_point)
    ver_line_length = np.linalg.norm(np.array(center_top) - np.array(center_bottom))
    ratio = hor_line_length / ver_line_length if ver_line_length != 0 else 0
    return ratio

def draw_button(frame, text, position, active):
    x, y = position
    w, h = 180, 40
    color = (0, 255, 0) if active else (200, 200, 200)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.putText(frame, text, (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def draw_calib_point(frame, pos):
    cv2.circle(frame, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

def on_mouse(event, x, y, flags, param):
    global active_eye
    if event == cv2.EVENT_LBUTTONDOWN:
        if WIDTH // 2 - 340 <= x <= WIDTH // 2 - 160 and 20 <= y <= 60:
            active_eye = 0
        elif WIDTH // 2 - 100 <= x <= WIDTH // 2 + 80 and 20 <= y <= 60:
            active_eye = 1
        elif WIDTH // 2 + 140 <= x <= WIDTH // 2 + 320 and 20 <= y <= 60:
            active_eye = 2

cv2.namedWindow("Gaze Calibration")
cv2.setMouseCallback("Gaze Calibration", on_mouse)

calibrating = False
print("Pressione 'c' para iniciar calibração automática")
print("Pressione 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    pred_x, pred_y = None, None
    color = (0, 255, 0)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks_for_calib = face_landmarks

        if active_eye == 0:
            features = extract_eye_features(face_landmarks, LEFT_IRIS, LEFT_EYE_BORDER)
            blink_ratio = get_blinking_ratio_mediapipe(face_landmarks, LEFT_EYE_BORDER)
        elif active_eye == 1:
            features = extract_eye_features(face_landmarks, RIGHT_IRIS, RIGHT_EYE_BORDER)
            blink_ratio = get_blinking_ratio_mediapipe(face_landmarks, RIGHT_EYE_BORDER)
        else:
            features_left = extract_eye_features(face_landmarks, LEFT_IRIS, LEFT_EYE_BORDER)
            features_right = extract_eye_features(face_landmarks, RIGHT_IRIS, RIGHT_EYE_BORDER)
            features = features_left + features_right
            blink_ratio_left = get_blinking_ratio_mediapipe(face_landmarks, LEFT_EYE_BORDER)
            blink_ratio_right = get_blinking_ratio_mediapipe(face_landmarks, RIGHT_EYE_BORDER)
            blink_ratio = (blink_ratio_left + blink_ratio_right) / 2

        if model_trained:
            pred = model.predict([features])[0]
            raw_x, raw_y = int(pred[0] * WIDTH), int(pred[1] * HEIGHT)

            # Aplica Filtro de Kalman
            pred_x, pred_y = kf.update(raw_x, raw_y)

            if blink_ratio > CLICK_THRESHOLD:
                color = (0, 255, 255)
                if not blinking:
                    blink_start_time = time.time()
                    blinking = True
                cv2.putText(frame, "Blinking Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            else:
                cv2.putText(frame, "Eyes Open", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if blinking and time.time() - blink_start_time >= BLINK_DURATION:
                    blinking = False
                    blink_start_time = None
                if not blinking:
                    color = (0, 255, 0)

            cv2.circle(frame, (int(pred_x), int(pred_y)), 20, color, -1)

    draw_button(frame, "Olho Esquerdo", (WIDTH // 2 - 340, 20), active_eye == 0)
    draw_button(frame, "Olho Direito", (WIDTH // 2 - 100, 20), active_eye == 1)
    draw_button(frame, "Ambos os Olhos", (WIDTH // 2 + 140, 20), active_eye == 2)

    if calibrating:
        point = calib_points[calib_idx]
        draw_calib_point(frame, point)

        if landmarks_for_calib is not None:
            frame_count += 1
            if frame_count == 1:
                temp_features = []

            if frame_count <= frames_per_point:
                temp_features.append(features)
            else:
                median_features = np.median(temp_features, axis=0).tolist()
                if np.std(temp_features, axis=0).max() < 0.05:
                    calib_features.append(median_features)
                    calib_targets.append([point[0] / WIDTH, point[1] / HEIGHT])

                calib_idx += 1
                frame_count = 0

                if calib_idx >= len(calib_points):
                    calibrating = False
                    model.fit(np.array(calib_features), np.array(calib_targets))
                    model_trained = True
    else:
        if not model_trained:
            cv2.putText(frame, "Pressione 'c' para iniciar calibração", (20, HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gaze Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and not calibrating:
        calib_features.clear()
        calib_targets.clear()
        calib_idx = 0
        frame_count = 0
        calibrating = True
    elif key == ord('g'):
        gain = max(0.5, gain - GAIN_STEP)
    elif key == ord('h'):
        gain = min(2.0, gain + GAIN_STEP)

cap.release()
cv2.destroyAllWindows()
