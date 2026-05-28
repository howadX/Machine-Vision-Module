from time import sleep
import glob, cv2, mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os as _os
import sys, select, tty, termios
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_SCRIPT_DIR))
dat_path = _os.path.join(_ROOT, 'models', 'face_detection') + '/'
sys.path.insert(0, _os.path.join(_ROOT, 'src', 'utils'))
from face_id import FaceID
face_rec_path = _os.path.join(_ROOT, 'models', 'face_recognition') + '/'
emotion_path = _os.path.join(_ROOT, 'models', 'emotion') + '/'
mp3_pic_path = _os.path.join(_ROOT, 'models', 'references') + '/'
assets_path = _os.path.join(_ROOT, 'assets', 'emotion_examples') + '/'
font_path = _os.path.join(_ROOT, 'assets', 'fonts', 'DejaVuSans.ttf')

def find_camera():
    """Ищет первую рабочую камеру через /dev/video* + CAP_V4L2."""
    devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")))
    for dev in devices:
        idx = int(dev.replace("/dev/video", ""))
        for method in ["path", "index"]:
            cam = cv2.VideoCapture(dev if method == "path" else idx, cv2.CAP_V4L2)
            if cam.isOpened():
                for _ in range(5):
                    ret, frame = cam.read()
                    if ret and frame is not None and frame.mean() > 10:
                        cam.release()
                        return dev
                cam.release()
    return None

# ── Чтение клавиш из терминала (работает без фокуса на окне) ──
_term_fd = sys.stdin.fileno()
_term_old = termios.tcgetattr(_term_fd)

def kb_init():
    """Переводит терминал в cbreak-режим для чтения одиночных нажатий."""
    tty.setcbreak(_term_fd)

def kb_restore():
    """Восстанавливает исходный режим терминала."""
    termios.tcsetattr(_term_fd, termios.TCSADRAIN, _term_old)

def kb_hit():
    """True если в stdin есть данные (клавиша нажата)."""
    return select.select([sys.stdin], [], [], 0)[0]

def kb_get():
    """Читает один символ из stdin (без ожидания)."""
    return sys.stdin.read(1) if kb_hit() else None

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# ── Фаза 1: идентификация ────────────────────────────
face_id_detector = FaceID()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cam_path = find_camera()
if cam_path is None:
    print("❌ Камера не найдена. Проверь: ls /dev/video*")
    exit(1)
video_capture = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
video_capture.set(3, 640)
video_capture.set(4, 480)
img_path = ['p_D.jpg','p_N.jpg','p_I.jpg','p_V.jpg']
fdi = {}
names = ['Даня','Никита','Белла','Вадим']
ind_u=0

for im, name in zip(img_path, names):
    img = cv2.imread(mp3_pic_path+im)
    emb = face_id_detector.get_embedding(img)
    if emb is not None:
        fdi[name] = emb
        print(f"  [Галерея] {name}: эмбеддинг {emb.shape}")
    else:
        print(f"  [Галерея] {name}: лицо не найдено")
ind_d,ind_u,ind_p,j=0,0,0,0
print("[Фаза 1] Идентификация. Нажми Q для перехода к эмоциям.")
kb_init()
while(1):
    ret, frame = video_capture.read()
    face_embedding = None

    if j % 3 == 0:
        # Каждый 3-й кадр: app.get() для bbox + embedding (ArcFace 512d)
        faces = face_id_detector.app.get(frame)
        if not faces:
            detections = []
        else:
            best = max(faces, key=lambda f: f.get("det_score", 0))
            bbox = best["bbox"]  # [x1, y1, x2, y2]
            face_embedding = np.asarray(best.get("embedding"), dtype=np.float32)
            detections = [(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))]
    else:
        # Остальные кадры: det_model.detect() только bbox
        bboxes, _ = face_id_detector.app.det_model.detect(frame)
        if bboxes.shape[0] == 0:
            detections = []
        else:
            detections = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes]

    if len(detections) == 0:
        if not ind_d:
            print('нет никого в кадре')
            ind_d,ind_p,ind_u=1,0,0
    if len(detections) > 0:
        d = detections[0]
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))

        if j % 3 == 0 and face_embedding is not None:
            if not ind_p:
                for name, ref_emb in fdi.items():
                    emb_norm = face_embedding / np.linalg.norm(face_embedding)
                    ref_norm = ref_emb / np.linalg.norm(ref_emb)
                    sim = float(np.dot(emb_norm, ref_norm))
                    if sim > 0.35:
                        print('Привет {}!'.format(name))
                        print('нажмите Q чтобы продолжить')
                        ind_p,ind_d,ind_u=1,0,1
            if not ind_u:
                print('Пользователь не распознан')
                ind_u,ind_d=1,0
    cv2.imshow('camera', image_resize(frame, height = 300))
    cv2.moveWindow('camera', 600,400)
    if cv2.waitKey(1) == -1 and kb_hit():
        ch = kb_get()
        if ch and ch.lower() == 'q':
            print('вы нажали q. Переход к эмоциям...')
            break
    j+=1
cv2.destroyAllWindows()

# ── Фаза 2: эмоции ───────────────────────────────────
nn=20
P_em=0.5
from emotion import EmotionClassifier
ec = EmotionClassifier(use_gpu=True)
font= ImageFont.truetype(font_path, 18)
emotions=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх']
pictures=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg']
print("[Фаза 2] Тренировка эмоций.")

for i, em in enumerate(emotions):
    tries = 0
    emotion_result = 0
    max_tries = 5
    while emotion_result < P_em and tries < max_tries:
        tries += 1
        frames_with_face = 0
        img = cv2.imread(assets_path + pictures[i])
        cv2.imshow('example', img)
        cv2.moveWindow('example', 150, 150)
        print("Покажите нам " + em)
        for j in range(nn):
            em_idx = 0  # всегда определён, даже если нет детекций
            ret, frame = video_capture.read()
            bboxes, _ = face_id_detector.app.det_model.detect(frame)
            if bboxes.shape[0] == 0:
                detections = []
            else:
                detections = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes]
            if len(detections) == 0:
                print(em_idx)
            for k, d in enumerate(detections):
                frames_with_face += 1
                if j % 1 == 0:
                    b = bboxes[k]
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        name_ru, conf = ec.predict(face_crop)
                        if name_ru in emotions:
                            em_idx = emotions.index(name_ru) + 1
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.multi_face_landmarks[0],
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))
                print(em_idx)
                if em_idx == i + 1:
                    emotion_result += 1
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((d[0], d[1] - 20), emotions[em_idx - 1], font=font, fill=(0, 0, 255))
                frame = np.array(img_pil)
            cv2.waitKey(10)
            cv2.imshow('camera', image_resize(frame, height=600))
            cv2.moveWindow('camera', 600, 150)
        emotion_result = emotion_result / nn
        print("[DEBUG] tries={}, emotion_result={:.2f}, frames_with_face={}".format(tries, emotion_result, frames_with_face))
        if emotion_result >= P_em:
            print("Молодец  " + "{0:.2f}".format(emotion_result * 100) + '%')
            sleep(2)
        elif tries >= max_tries:
            print("Пропускаем {} — лицо не найдено".format(em))
            sleep(2)
        else:
            print("Попробуте еще " + "{0:.2f}".format(emotion_result * 100) + '%')
            sleep(2)
video_capture.release()
kb_restore()
cv2.destroyAllWindows()
