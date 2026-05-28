from scipy.spatial import distance
from time import sleep
import glob, cv2, dlib
from PIL import ImageFont, ImageDraw, Image
import pickle
import numpy as np
import random
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
predictor = dlib.shape_predictor(dat_path+'shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(face_rec_path+'dlib_face_recognition_resnet_model_v1.dat')
cam_path = find_camera()
if cam_path is None:
    print("❌ Камера не найдена. Проверь: ls /dev/video*")
    exit(1)
video_capture = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
video_capture.set(3, 640)
video_capture.set(4, 480)
img_path = ['p_D.jpg','p_N.jpg','p_I.jpg','p_V.jpg']
fdi = []
names = ['Даня','Никита','Белла','Вадим']
ind_u=0

for im in img_path:
    img = cv2.imread(mp3_pic_path+im)
    bboxes, _ = face_id_detector.app.det_model.detect(img)
    if bboxes.shape[0] == 0:
        detections = []
    else:
        detections = [dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes]
    for k,d in enumerate(detections):
        shape = predictor(img, d)
        face_descriptor_img = facerec.compute_face_descriptor(img, shape)
        fdi.append(face_descriptor_img)
ind_d,ind_u,ind_p,j=0,0,0,0
print("[Фаза 1] Идентификация. Нажми Q для перехода к эмоциям.")
kb_init()
while(1):
    ret, frame = video_capture.read()
    bboxes, _ = face_id_detector.app.det_model.detect(frame)
    if bboxes.shape[0] == 0:
        detections = []
    else:
        detections = [dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes]
    face_descriptor_frame =0
    if len(detections) == 0:
        if not ind_d:
            print('нет никого в кадре')
            ind_d,ind_p,ind_u=1,0,0
    for k,d in enumerate(detections):
        if j%3==0:
            shape = predictor(frame, d)
            face_descriptor_frame = facerec.compute_face_descriptor(frame, shape)
        cv2.rectangle(frame, (detections[0].left(), detections[0].top()), (detections[0].right(), detections[0].bottom()), (0, 0, 255), 1)
        for k in range(0,17):
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,0,255), 1)
        for k in range(18,26):
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,255,255), 1)
        for k in range(27,36):
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0,255,0), 1)
        for k in range(36,48):
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255,0,0), 1)
        for k in range(49,68):
            cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0,0,255), 1)
        cv2.circle(frame, (shape.part(30).x, shape.part(30).y), 1, (0,255,255), 1)

        if j%3==0:
            if not ind_p:
                for c,fd in enumerate(fdi):
                    q = distance.euclidean(fd,face_descriptor_frame)
                    if q<0.6:
                        print('Привет {}!'.format(names[c]))
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
# detector уже создан глобально — используем face_id_detector
predictor = dlib.shape_predictor(dat_path+'shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1(face_rec_path+'dlib_face_recognition_resnet_model_v1.dat')
font= ImageFont.truetype(font_path, 18)
emotions=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх']
pictures=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg']
f=open(emotion_path+'svm_dat.dat','rb')
clf=pickle.load(f)
f.close
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
                detections = [dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bboxes]
            if len(detections) == 0:
                print(em_idx)
            for k, d in enumerate(detections):
                frames_with_face += 1
                if j % 1 == 0:
                    shape = predictor(frame, d)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    dat = []
                    dat.append(face_descriptor)
                    em_n = clf.predict(dat)
                    em_idx = int(em_n[0])
                cv2.rectangle(frame, (detections[0].left(), detections[0].top()), (detections[0].right(), detections[0].bottom()), (0, 0, 255), 1)
                for k in range(0, 17):
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255, 0, 255), 1)
                for k in range(18, 26):
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255, 255, 255), 1)
                for k in range(27, 36):
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0, 255, 0), 1)
                for k in range(36, 48):
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (255, 0, 0), 1)
                for k in range(49, 68):
                    cv2.circle(frame, (shape.part(k).x, shape.part(k).y), 1, (0, 0, 255), 1)
                cv2.circle(frame, (shape.part(30).x, shape.part(30).y), 1, (0, 255, 255), 1)
                print(em_idx)
                if em_idx == i + 1:
                    emotion_result += 1
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((detections[0].left(), detections[0].top() - 20), emotions[em_idx - 1], font=font, fill=(0, 0, 255))
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
