"""
人脸检测模块：RobustFaceDetector
优先使用 OpenCV DNN，回退到 Haar Cascade
"""

import os
import cv2
import numpy as np


def _load_cascade_safe(xml_path):
    """
    安全加载 Haar Cascade：先尝试直接路径，失败则用字节流。
    """
    # 先尝试直接加载（英文路径可用）
    detector = cv2.CascadeClassifier(xml_path)
    if not detector.empty():
        return detector

    # 直接加载失败（中文路径问题），用字节流方式
    tmp_path = None
    try:
        with open(xml_path, "rb") as f:
            xml_bytes = f.read()
        # 写入临时文件（用英文路径）再加载
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            tmp.write(xml_bytes)
            tmp_path = tmp.name
        detector = cv2.CascadeClassifier(tmp_path)
        if not detector.empty():
            return detector
    except Exception:
        pass
    finally:
        # 确保临时文件被清理，即使程序异常也不会残留
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return None


def _load_dnn_safe(proto_path, model_path):
    """
    安全加载 DNN 模型：先尝试直接路径，失败则用字节流。
    """
    # 先尝试直接加载
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        return net
    except Exception:
        pass

    # 中文路径问题：用字节流
    try:
        with open(proto_path, "rb") as f:
            proto_bytes = f.read()
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        net = cv2.dnn.readNetFromCaffe(proto_bytes, model_bytes)
        return net
    except Exception:
        pass

    return None


class RobustFaceDetector:
    """
    鲁棒的人脸检测器：
    - 优先使用 OpenCV DNN (res10_300x300_ssd)，精度更高
    - 回退使用 Haar Cascade，开箱即用
    - 支持中文路径（通过字节流加载绕过 OpenCV 路径编码问题）
    """

    # DNN 模型文件名
    DNN_PROTO = "deploy.prototxt"
    DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
    HAAR_FILE = "haarcascade_frontalface_default.xml"

    def __init__(self, confidence_threshold=0.5, model_dir="models"):
        self.confidence_threshold = confidence_threshold
        self.model_dir = model_dir
        self.use_dnn = False
        self.detector = None

        # 尝试加载 DNN
        proto_path = os.path.join(model_dir, self.DNN_PROTO)
        model_path = os.path.join(model_dir, self.DNN_MODEL)

        if os.path.isfile(proto_path) and os.path.isfile(model_path):
            net = _load_dnn_safe(proto_path, model_path)
            if net is not None:
                self.detector = net
                self.use_dnn = True
                print("[检测] 使用 OpenCV DNN 检测器")
                return
            else:
                print("[检测] DNN 加载失败，尝试 Haar Cascade...")

        # 回退到 Haar Cascade（项目 models/ 目录）
        haar_path = os.path.join(model_dir, self.HAAR_FILE)
        if os.path.isfile(haar_path):
            detector = _load_cascade_safe(haar_path)
            if detector is not None:
                self.detector = detector
                print("[检测] 使用 Haar Cascade 检测器")
                return

        # 尝试从 OpenCV 内置路径加载
        cv2_base = os.path.dirname(cv2.__file__)
        haar_builtin = os.path.join(cv2_base, "data", self.HAAR_FILE)
        if os.path.isfile(haar_builtin):
            detector = _load_cascade_safe(haar_builtin)
            if detector is not None:
                self.detector = detector
                print("[检测] 使用 OpenCV 内置 Haar Cascade 检测器")
                return

        raise RuntimeError(
            "无法加载任何人脸检测模型！\n"
            "请将以下文件放入 models/ 目录：\n"
            "  - deploy.prototxt\n"
            "  - res10_300x300_ssd_iter_140000.caffemodel\n"
            "或者确保 Haar Cascade 文件可用。"
        )

    def detect(self, gray, bgr=None):
        """
        检测人脸，返回矩形列表 [(x, y, w, h), ...]。

        Parameters:
            gray: 灰度图像
            bgr: BGR 图像（DNN 模式需要，灰度模式可为 None）

        Returns:
            faces: 人脸矩形列表
        """
        if self.use_dnn:
            return self._detect_dnn(bgr if bgr is not None else gray)
        else:
            return self._detect_haar(gray)

    def _detect_dnn(self, frame):
        """使用 DNN 检测"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
        self.detector.setInput(blob)
        detections = self.detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                # 确保坐标有效
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_haar(self, gray):
        """使用 Haar Cascade 检测"""
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [tuple(f) for f in faces]


if __name__ == "__main__":
    # 快速测试：打开摄像头检测人脸
    detector = RobustFaceDetector()
    cap = cv2.VideoCapture(0)
    print("按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(gray, frame)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
