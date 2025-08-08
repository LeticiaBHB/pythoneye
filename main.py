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
WIDTH, HEIGHT = 1366, 768  # Resolução grande
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Posição da bolinha
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
smoothing_factor = 0.15
active_eye = 0  # 0 = esquerdo, 1 = direito

# Índices landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_BORDER = [33, 133, 159, 145, 153, 154]
RIGHT_EYE_BORDER = [362, 263, 386, 374, 380, 381]

# Função para desenhar botão moderno
def draw_button(frame, text, position, is_active):
    x, y = position
    color_bg = (0, 200, 0) if is_active else (60, 60, 60)
    color_border = (255, 255, 255)

    # Fundo do botão com bordas arredondadas simuladas
    cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), color_bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + BUTTON_WIDTH, y + BUTTON_HEIGHT), color_border, 2, cv2.LINE_AA)

    # Texto centralizado
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = x + (BUTTON_WIDTH - text_size[0]) // 2
    text_y = y + (BUTTON_HEIGHT + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Detecta clique no botão
def check_button_click(mouse_x, mouse_y, button_pos):
    x, y = button_pos
    return x <= mouse_x <= x + BUTTON_WIDTH and y <= mouse_y <= y + BUTTON_HEIGHT

# Variáveis para clique do mouse
mouse_x, mouse_y = 0, 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

cv2.namedWindow("Controle com os Olhos", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Controle com os Olhos", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Controle com os Olhos", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Botões centralizados no topo
    draw_button(frame, "Olho Esquerdo", (WIDTH // 2 - 220, 20), active_eye == 0)
    draw_button(frame, "Olho Direito", (WIDTH // 2 + 20, 20), active_eye == 1)

    # Detecta clique do mouse
    if mouse_clicked:
        if check_button_click(mouse_x, mouse_y, (WIDTH // 2 - 220, 20)):
            active_eye = 0
        elif check_button_click(mouse_x, mouse_y, (WIDTH // 2 + 20, 20)):
            active_eye = 1
        mouse_clicked = False

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Escolhe o olho ativo
        if active_eye == 0:
            iris_landmark = face_landmarks.landmark[LEFT_IRIS[0]]
            eye_border = [face_landmarks.landmark[i] for i in LEFT_EYE_BORDER]
        else:
            iris_landmark = face_landmarks.landmark[RIGHT_IRIS[0]]
            eye_border = [face_landmarks.landmark[i] for i in RIGHT_EYE_BORDER]

        iris_x = int(iris_landmark.x * w)
        iris_y = int(iris_landmark.y * h)

        eye_min_x = min([p.x for p in eye_border]) * w
        eye_max_x = max([p.x for p in eye_border]) * w
        eye_min_y = min([p.y for p in eye_border]) * h
        eye_max_y = max([p.y for p in eye_border]) * h

        norm_x = (iris_x - eye_min_x) / (eye_max_x - eye_min_x)
        norm_y = (iris_y - eye_min_y) / (eye_max_y - eye_min_y)

        target_x = int(norm_x * WIDTH)
        target_y = int(norm_y * HEIGHT)

        ball_x = int(ball_x * (1 - smoothing_factor) + target_x * smoothing_factor)
        ball_y = int(ball_y * (1 - smoothing_factor) + target_y * smoothing_factor)

        ball_x = np.clip(ball_x, 0, WIDTH)
        ball_y = np.clip(ball_y, 0, HEIGHT)

        cv2.circle(frame, (iris_x, iris_y), 3, (0, 255, 255), -1)

    cv2.circle(frame, (ball_x, ball_y), 20, (255, 0, 0), -1)

    cv2.imshow("Controle com os Olhos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



