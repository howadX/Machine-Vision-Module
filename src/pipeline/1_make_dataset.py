import cv2
import time
import glob
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_SCRIPT_DIR))
dat_path = _os.path.join(_ROOT, 'models', 'face_detection') + '/'

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

cam_path = find_camera()
if cam_path is None:
    print("❌ Камера не найдена. Проверь: ls /dev/video*")
    exit(1)
cam = cv2.VideoCapture(cam_path, cv2.CAP_V4L2) # захватываем изображение
cam.set(3, 800) # устанавливаем высоту кадра
cam.set(4, 600) # устанавливаем ширину кадра
face_detector = cv2.CascadeClassifier(dat_path+'haarcascade_frontalface_default.xml') # Установим классификатор лица, предоставляемый библиотекой OpenCV
face_id = input('\n Введите номер эмоции и нажмите enter ==>  ') # Для каждой эмоции, устанавливаем свой id
print("\t [INFO] Инициализация лица. Смотрите в камеру и ждите...")
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Переводим изображение в оттенки серого
    '''faces = face_detector.detectMultiScale(gray, 1.3, 5) # Выделяем лицо в прямоугольную область
    for (x,y,w,h) in faces:
        count+=1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)'''
    cv2.imwrite("Img/page"+str(count), gray) # Сохраняем захваченное изображение в папку dataset "dataset/Em_{}/EM_{}_{}.jpg".format(face_id, face_id, int(time.time()*1e9)), gray[y:y+h,x:x+w]
    cv2.imshow('image', img)
    if cv2.waitKey(300) & 0xff == 27:
        break # Для завершения работы программы нажмите «ESC»
    elif count >= 30:
        break # Выполнение программы заканчивается после того, как сделает определённое количество кадров (захватит изображений лица)
    print(count)

print("\n [INFO] Закрытие программы")
cam.release()
cv2.destroyAllWindows() # Заканчиваем видеопоток и закрываем окна
