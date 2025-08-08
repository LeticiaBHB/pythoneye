import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LinearRegression

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

WIDTH, HEIGHT = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

LEFT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE_BORDER = [33, 133, 159, 145, 153, 154]

RIGHT_IRIS = [473, 474, 475, 476, 477]
RIGHT_EYE_BORDER = [263, 362, 386, 374, 380, 381]

calib_features = []
calib_targets = []

model_trained = False
model = LinearRegression()

active_eye = 0  # 0 = olho esquerdo, 1 = olho direito
landmarks_for_calib = None

# GRADE 5x5 para mais pontos, mais precisão
grid_x = np.linspace(0.1, 0.9, 5)
grid_y = np.linspace(0.1, 0.9, 5)
calib_points = [(x * WIDTH, y * HEIGHT) for y in grid_y for x in grid_x]

calib_idx = 0
frames_per_point = 50  # Mais frames para maior média
frame_count = 0
temp_features = []


def extract_eye_features(landmarks, iris_idxs, eye_border_idxs):
    iris_points = [landmarks[i] for i in iris_idxs]
    eye_border_points = [landmarks[i] for i in eye_border_idxs]

    iris_x = np.mean([p.x for p in iris_points])
    iris_y = np.mean([p.y for p in iris_points])

    min_x = min([p.x for p in eye_border_points])
    max_x = max([p.x for p in eye_border_points])
    min_y = min([p.y for p in eye_border_points])
    max_y = max([p.y for p in eye_border_points])

    # Normalização melhor: evitar divisão por zero
    norm_x = (iris_x - min_x) / (max_x - min_x) if (max_x - min_x) != 0 else 0
    norm_y = (iris_y - min_y) / (max_y - min_y) if (max_y - min_y) != 0 else 0

    # Distâncias relativas dentro do olho para mais robustez
    width = max_x - min_x
    height = max_y - min_y

    # Distância da íris ao centro do olho (normalizado)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    dist_x = (iris_x - center_x) / width if width != 0 else 0
    dist_y = (iris_y - center_y) / height if height != 0 else 0

    return [norm_x, norm_y, dist_x, dist_y]


def draw_button(frame, text, position, active):
    x, y = position
    w, h = 180, 40
    color = (0, 255, 0) if active else (200, 200, 200)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.putText(frame, text, (x + 10, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


def draw_calib_point(frame, pos):
    cv2.circle(frame, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)


def on_mouse(event, x, y, flags, param):
    global active_eye
    if event == cv2.EVENT_LBUTTONDOWN:
        if WIDTH // 2 - 220 <= x <= WIDTH // 2 - 220 + 180 and 20 <= y <= 60:
            active_eye = 0
            print("Olho esquerdo selecionado")
        elif WIDTH // 2 + 20 <= x <= WIDTH // 2 + 20 + 180 and 20 <= y <= 60:
            active_eye = 1
            print("Olho direito selecionado")


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

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks_for_calib = face_landmarks

        if active_eye == 0:
            features = extract_eye_features(face_landmarks, LEFT_IRIS, LEFT_EYE_BORDER)
            iris_points = LEFT_IRIS
        else:
            features = extract_eye_features(face_landmarks, RIGHT_IRIS, RIGHT_EYE_BORDER)
            iris_points = RIGHT_IRIS

        if model_trained:
            pred = model.predict([features])[0]
            pred_x = int(pred[0] * WIDTH)
            pred_y = int(pred[1] * HEIGHT)
            cv2.circle(frame, (pred_x, pred_y), 20, (0, 255, 0), -1)

        for idx in iris_points:
            lm = face_landmarks[idx]
            cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    else:
        landmarks_for_calib = None

    draw_button(frame, "Olho Esquerdo", (WIDTH // 2 - 220, 20), active_eye == 0)
    draw_button(frame, "Olho Direito", (WIDTH // 2 + 20, 20), active_eye == 1)

    if calibrating:
        point = calib_points[calib_idx]
        draw_calib_point(frame, point)
        cv2.putText(frame, f"Olhe para o ponto {calib_idx + 1}/{len(calib_points)}",
                    (20, HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if landmarks_for_calib is not None:
            frame_count += 1
            if frame_count == 1:
                temp_features = []

            if frame_count <= frames_per_point:
                if active_eye == 0:
                    f = extract_eye_features(landmarks_for_calib, LEFT_IRIS, LEFT_EYE_BORDER)
                else:
                    f = extract_eye_features(landmarks_for_calib, RIGHT_IRIS, RIGHT_EYE_BORDER)
                temp_features.append(f)
            else:
                # Mediana para eliminar outliers
                median_features = np.median(temp_features, axis=0).tolist()

                # Verificação simples de estabilidade (desvio)
                std_dev = np.std(temp_features, axis=0)
                if np.any(std_dev > 0.05):  # limiar, ajustar se quiser
                    print(f"Aviso: alta variação nas amostras do ponto {calib_idx + 1}, descartando")
                else:
                    calib_features.append(median_features)
                    calib_targets.append([point[0] / WIDTH, point[1] / HEIGHT])
                    print(f"Calibrado ponto {calib_idx + 1} com features {median_features}")

                calib_idx += 1
                frame_count = 0
                temp_features = []

                if calib_idx >= len(calib_points):
                    calibrating = False
                    calib_idx = 0
                    print("Treinando modelo com dados calibrados...")
                    X = np.array(calib_features)
                    y = np.array(calib_targets)
                    model.fit(X, y)
                    model_trained = True
                    print("Modelo treinado com sucesso!")

    else:
        cv2.putText(frame, "Pressione 'c' para iniciar calibração automática",
                    (20, HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Olho ativo: {'Esquerdo' if active_eye == 0 else 'Direito'}",
                (WIDTH - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gaze Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and not calibrating:
        calib_features = []
        calib_targets = []
        calib_idx = 0
        frame_count = 0
        calibrating = True
        print("Iniciando calibração automática... Olhe para os pontos!")

cap.release()
cv2.destroyAllWindows()
