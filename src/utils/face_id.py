"""
face_id.py — Модуль распознавания лиц на базе ArcFace (InsightFace).
Заменяет dlib HOG + ResNet-128d.

Модель buffalo_l (SCRFD детектор + ArcFace 512d):
  - Детекция:  SCRFD (∼96% точности, профиль/маска/очки)
  - Landmarks:  5 точек (выравнивание)
  - Эмбеддинг:  ArcFace 512d (99.83% LFW)

Использование:
    from face_id import FaceID
    fid = FaceID()
    embedding = fid.get_embedding(image_bgr)        # → np.array(512,) или None
    name, dist = fid.identify(embedding, gallery)   # → (str, float)
"""
import numpy as np
import os
import sys
import glob as _glob
import ctypes

# ── Авто-подхват CUDA-библиотек из nvidia-* pip-пакетов ──
def _setup_cuda_libs():
    """Предзагружает CUDA-библиотеки через ctypes, чтобы onnxruntime их нашёл."""
    # face_id.py лежит в src/utils/ → 2 уровня вверх = корень проекта, + venv/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    venv_root = os.path.join(project_root, "venv")
    nvidia_base = os.path.join(venv_root, "lib", "python3.14", "site-packages", "nvidia")
    if not os.path.isdir(nvidia_base):
        return
    # Предзагружаем ключевые .so
    libs = []
    for lib_dir in _glob.iglob(os.path.join(nvidia_base, "**/lib"), recursive=True):
        for so in _glob.glob(os.path.join(lib_dir, "*.so*")):
            libs.append(so)
    # Порядок важен: сначала cublas, cudnn, потом остальные
    priority = ["cublas", "cudnn", "curand", "cufft", "cusolver", "cusparse", "nvrtc", "nvjitlink", "cuda_runtime"]
    def _key(path):
        name = os.path.basename(path)
        for i, p in enumerate(priority):
            if p in name.lower().replace("_", "").replace("-", ""):
                return i
        return 99
    libs.sort(key=_key)
    for lib in libs:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # не все .so можно загрузить как библиотеку

_setup_cuda_libs()

from insightface.app import FaceAnalysis

# Путь для кэша моделей (~300MB, скачается один раз)
MODEL_CACHE = os.path.expanduser("~/.insightface")


class FaceID:
    """Обёртка над InsightFace ArcFace для детекции и распознавания лиц."""

    def __init__(self, model_name: str = "buffalo_l", use_gpu: bool = True):
        """
        Args:
            model_name: buffalo_l (SCRFD + ArcFace) — лучший баланс скорость/качество.
                        Также доступны: buffalo_m, buffalo_s, antelopev2.
            use_gpu: True = CUDA (твоя RTX 4070 Ti), False = CPU (onnxruntime).
        """
        self.model_name = model_name
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=model_name,
            root=MODEL_CACHE,
            providers=providers,
        )
        # Подготовка: детекция + распознавание
        self.app.prepare(
            ctx_id=0 if use_gpu else -1,  # GPU device 0 или CPU
            det_size=(640, 480),
        )
        self._embedding_dim = 512  # ArcFace

    @property
    def dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """
        Возвращает 512-мерный эмбеддинг самого уверенного лица на изображении.
        Если лиц не найдено — None.

        Args:
            image_bgr: изображение в формате BGR (H, W, 3), как из cv2.imread().

        Returns:
            np.ndarray shape=(512,) или None.
        """
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        # Берём самое уверенное лицо
        best = max(faces, key=lambda f: f.get("det_score", 0))
        emb = best.get("embedding")
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)

    def get_face_info(self, image_bgr: np.ndarray) -> dict | None:
        """
        Возвращает полную информацию о лице: bbox, 5 landmarks, эмбеддинг, уверенность.
        """
        faces = self.app.get(image_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.get("det_score", 0))
        return {
            "bbox": best["bbox"].tolist(),          # [x1, y1, x2, y2]
            "landmarks": best.get("kps", None),      # 5 точек (глаза, нос, губы)
            "embedding": np.asarray(best["embedding"], dtype=np.float32),
            "det_score": float(best.get("det_score", 0)),
        }

    def identify(
        self,
        embedding: np.ndarray,
        gallery: dict[str, np.ndarray],
        threshold: float = 0.35,
    ) -> tuple[str | None, float]:
        """
        Идентифицирует человека по эмбеддингу.

        Args:
            embedding: эмбеддинг живого лица (512d).
            gallery: словарь {имя: эмбеддинг}.
            threshold: порог косинусного сходства (0..1).
                       0.35 = уверенное совпадение, 0.25 = слабое.

        Returns:
            (имя, similarity_score) или (None, max_score).
        """
        best_name = None
        best_sim = -1.0

        emb_norm = embedding / np.linalg.norm(embedding)

        for name, ref_emb in gallery.items():
            ref_norm = ref_emb / np.linalg.norm(ref_emb)
            sim = float(np.dot(emb_norm, ref_norm))  # косинусное сходство
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < threshold:
            return None, best_sim
        return best_name, best_sim

    def build_gallery(
        self, image_paths: dict[str, str]
    ) -> dict[str, np.ndarray]:
        """
        Строит галерею эталонов: {имя: путь} → {имя: эмбеддинг}.

        Args:
            image_paths: {имя: путь_к_фото}

        Returns:
            {имя: np.array(512,)} — только для успешно распознанных.
        """
        gallery = {}
        for name, path in image_paths.items():
            img = _read_bgr(path)
            if img is None:
                print(f"  [WARN] Галерея: {name} — не удалось прочитать {path}")
                continue
            emb = self.get_embedding(img)
            if emb is None:
                print(f"  [WARN] Галерея: {name} — лицо не найдено в {path}")
                continue
            gallery[name] = emb
        return gallery


def _read_bgr(path: str) -> np.ndarray | None:
    """Читает изображение в BGR. Возвращает None при ошибке."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    return img
