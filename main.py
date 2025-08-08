"""
Controle com os Olhos — Versão melhorada

O que eu alterei:
- Melhor precisão: uso da média dos pontos da íris (vários landmarks) para calcular o centro do olhar
  e normalização pelo contorno do olho. Melhorei também o "smoothing" para resposta mais estável.
- Removi o botão/calibração (pedido: sem botão calibrar).
- Piscar detectado por EAR (Eye Aspect Ratio). Ao detectar um piscar, a bolinha fica VERMELHA por 10s
  e depois volta para AZUL.
- Pequenas mensagens de debug (EAR) e desenho do contorno/íris para ajudar na depuração.

Dependências: opencv, mediapipe, numpy
Instalação: pip install opencv-python mediapipe numpy

Pressione ESC para sair.
"""

import time
import math
import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# Constantes da tela
WIDTH, HEIGHT = 1366, 768  # Ajuste conforme necessário
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Posição da bolinha
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
smoothing_factor = 0.12  # menor -> mais responsivo; maior -> mais suave
active_eye = 0  # 0 = esquerdo, 1 = direito

# Índices landmarks (MediaPipe Face Mesh)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
# Conjunto de 6 pontos usados para EAR (índices comumente usados no Face Mesh)
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
# Usaremos também um contorno mais amplo para normalização (pode ser ajustado)
LEFT_EYE_BORDER = [33, 133, 159, 145, 153, 154, 155, 133]
RIGHT_EYE_BORDER = [362, 263, 386, 374, 380, 381, 382, 263]

# Blink detection
EAR_THRESHOLD = 0.23
EAR_CONSEC_FRAMES = 3
blink_counter = 0
blink_end_time = 0.0

# Mouse / UI
mouse_x, mouse_y = 0, 0
mouse_clicked = False

# Função para desenhar botão moderno
def draw_button(frame, text, position, is_active):
    x, y = position
    color_bg = (0, 200, 0) if is_active else (60, 60, 60)
    color_border = (255, 255, 255)

    cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), color_bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), color_border, 2, cv2.LINE_AA)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = x + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = y + (BUTTON_HEIGHT + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Detecta clique no botão
def check_button_click(mouse_x, mouse_y, button_pos):
    x, y = button_pos
    return x <= mouse_x <= x + BUTTON_WIDTH and y <= mouse_y <= y + BUTTON_HEIGHT

# Callback do mouse (somente para trocar olho ativo)
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

cv2.namedWindow("Controle com os Olhos", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Controle com os Olhos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Controle com os Olhos", mouse_callback)

# Calcula EAR a partir de 6 landmarks
def calculate_EAR(landmarks, eye_indices, w, h):
    # p1..p6 conforme fórmula EAR
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    p1 = np.array(coords[0])
    p2 = np.array(coords[1])
    p3 = np.array(coords[2])
    p4 = np.array(coords[3])
    p5 = np.array(coords[4])
    p6 = np.array(coords[5])

    # Distâncias
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    hor = np.linalg.norm(p1 - p4)

    if hor == 0:
        return 1.0
    ear = (vert1 + vert2) / (2.0 * hor)
    return ear

# Função para obter centro da íris pela média dos pontos
def iris_center(landmarks, iris_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in iris_indices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(np.mean(xs)), int(np.mean(ys))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Botões centralizados no topo (somente dois: olho esquerdo/direito)
    draw_button(frame, "Olho Esquerdo", (WIDTH // 2 - 220, 20), active_eye == 0)
    draw_button(frame, "Olho Direito", (WIDTH // 2 + 20, 20), active_eye == 1)

    # Detecta clique do mouse para trocar olho ativo
    if mouse_clicked:
        if check_button_click(mouse_x, mouse_y, (WIDTH // 2 - 220, 20)):
            active_eye = 0
        elif check_button_click(mouse_x, mouse_y, (WIDTH // 2 + 20, 20)):
            active_eye = 1
        mouse_clicked = False

    ball_color = (255, 0, 0)  # azul por padrão (BGR)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Calcula centros das íris (média dos pontos disponíveis)
        left_iris_cx, left_iris_cy = iris_center(face_landmarks.landmark, LEFT_IRIS, w, h)
        right_iris_cx, right_iris_cy = iris_center(face_landmarks.landmark, RIGHT_IRIS, w, h)

        # Escolhe o olho ativo para controlar a bolinha
        if active_eye == 0:
            iris_x, iris_y = left_iris_cx, left_iris_cy
            eye_border_indices = LEFT_EYE_BORDER
        else:
            iris_x, iris_y = right_iris_cx, right_iris_cy
            eye_border_indices = RIGHT_EYE_BORDER

        # Contorno do olho usado para normalização
        eye_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in eye_border_indices]
        eye_min_x = min([p[0] for p in eye_pts])
        eye_max_x = max([p[0] for p in eye_pts])
        eye_min_y = min([p[1] for p in eye_pts])
        eye_max_y = max([p[1] for p in eye_pts])

        # Proteção contra divisão por zero
        denom_x = (eye_max_x - eye_min_x) if (eye_max_x - eye_min_x) != 0 else 1
        denom_y = (eye_max_y - eye_min_y) if (eye_max_y - eye_min_y) != 0 else 1

        norm_x = (iris_x - eye_min_x) / denom_x
        norm_y = (iris_y - eye_min_y) / denom_y

        # Aplica alguma sensibilidade (ajuste aqui se quiser mais alcance)
        sens_x = 1.0
        sens_y = 1.0

        target_x = int(np.clip(norm_x * WIDTH * sens_x, 0, WIDTH))
        target_y = int(np.clip(norm_y * HEIGHT * sens_y, 0, HEIGHT))

        # Suaviza movimento
        ball_x = int(ball_x * (1 - smoothing_factor) + target_x * smoothing_factor)
        ball_y = int(ball_y * (1 - smoothing_factor) + target_y * smoothing_factor)

        # Desenhos auxiliares
        cv2.circle(frame, (iris_x, iris_y), 3, (0, 255, 255), -1)
        for p in eye_pts:
            cv2.circle(frame, p, 1, (200, 200, 200), -1)

        # DETECÇÃO DE PISCAR (EAR)
        left_ear = calculate_EAR(face_landmarks.landmark, LEFT_EYE_EAR, w, h)
        right_ear = calculate_EAR(face_landmarks.landmark, RIGHT_EYE_EAR, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Debug: mostra EAR
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Contador de frames consecutivos com EAR baixo
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= EAR_CONSEC_FRAMES:
                # Piscar detectado -> ativa vermelho por 10 segundos
                blink_end_time = time.time() + 10.0
            blink_counter = 0

        # Se dentro do período de vermelho, altera cor
        if time.time() < blink_end_time:
            ball_color = (0, 0, 255)  # VERMELHO
            cv2.putText(frame, "PISCAR: VERMELHO (tempo restante: {:.0f}s)".format(max(0, blink_end_time - time.time())),
                        (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Desenha a bolinha (usa a cor determinada acima)
    ball_x = int(np.clip(ball_x, 0, WIDTH))
    ball_y = int(np.clip(ball_y, 0, HEIGHT))
    cv2.circle(frame, (ball_x, ball_y), 20, ball_color, -1)

    cv2.imshow("Controle com os Olhos", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
