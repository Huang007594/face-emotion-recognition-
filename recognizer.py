"""
识别推理模块：EmotionRecognizer
包含预处理、预测、滑动窗口平滑、UI 绘制
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from detector import RobustFaceDetector
from data import EMOTION_LABELS


# 表情对应的 emoji（终端显示用）
EMOJI_MAP = {
    0: "😠",  # Angry
    1: "🤮",  # Disgust
    2: "😨",  # Fear
    3: "😄",  # Happy
    4: "😢",  # Sad
    5: "😲",  # Surprise
    6: "😐",  # Neutral
}

# 表情对应的颜色（BGR）
COLOR_MAP = {
    0: (0, 0, 255),    # Angry - 红
    1: (0, 100, 255),  # Disgust - 橙
    2: (128, 0, 128),  # Fear - 紫
    3: (0, 255, 0),    # Happy - 绿
    4: (255, 0, 0),    # Sad - 蓝
    5: (0, 255, 255),  # Surprise - 黄
    6: (200, 200, 200),# Neutral - 灰
}


class EmotionRecognizer:
    """
    人脸表情识别器：集成检测 + 预处理 + 预测 + 平滑 + UI 绘制
    """

    def __init__(self, model_path, smooth_window=10, confidence_threshold=0.5):
        """
        Parameters:
            model_path: 训练好的 Keras 模型路径 (.h5)
            smooth_window: 滑动窗口长度
            confidence_threshold: 人脸检测置信度阈值
        """
        # 加载表情识别模型
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        print(f"[识别] 加载模型: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        # 初始化人脸检测器
        self.face_detector = RobustFaceDetector(
            confidence_threshold=confidence_threshold
        )

        # 滑动窗口：每个人脸维护一个 deque
        self.smooth_window = smooth_window
        self.prediction_history = {}  # face_id -> deque of predictions

        # FPS 计算
        self.frame_times = deque(maxlen=30)

    def preprocess_face(self, gray, face_rect):
        """
        从灰度图中裁剪人脸并预处理为模型输入。

        Parameters:
            gray: 灰度图
            face_rect: (x, y, w, h)

        Returns:
            preprocessed: (1, 48, 48, 1) 归一化张量
        """
        x, y, w, h = face_rect
        # 边界保护
        x, y = max(0, x), max(0, y)
        face = gray[y : y + h, x : x + w]
        if face.size == 0:
            return None

        # 缩放到 48x48，归一化
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)
        return face

    def predict(self, preprocessed_face):
        """
        单次前向传播，返回概率分布。
        """
        probs = self.model.predict(preprocessed_face, verbose=0)[0]
        return probs

    def smooth_predict(self, face_id, probs):
        """
        滑动窗口平均：对多帧预测结果取平均，抑制抖动。

        Parameters:
            face_id: 人脸标识（用于区分多人）
            probs: 当前帧的概率分布

        Returns:
            smoothed_probs: 平滑后的概率分布
        """
        if face_id not in self.prediction_history:
            self.prediction_history[face_id] = deque(maxlen=self.smooth_window)

        self.prediction_history[face_id].append(probs)

        # 取平均
        history = self.prediction_history[face_id]
        smoothed = np.mean(history, axis=0)
        return smoothed

    def draw_ui(self, frame, face_rect, emotion_idx, probs, fps=None):
        """
        在帧上绘制识别结果：人脸框、表情文字、置信度条。

        Parameters:
            frame: BGR 图像
            face_rect: (x, y, w, h)
            emotion_idx: 预测的表情索引
            probs: 概率分布
            fps: 当前帧率
        """
        x, y, w, h = face_rect
        color = COLOR_MAP.get(emotion_idx, (255, 255, 255))
        label = EMOTION_LABELS[emotion_idx]
        emoji = EMOJI_MAP[emotion_idx]
        confidence = probs[emotion_idx]

        # 人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 表情标签背景
        text = f"{emoji} {label} {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), color, -1)
        cv2.putText(frame, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)

        # 置信度条
        bar_x = x + w + 5
        bar_width = 100
        bar_height = 8
        for i, prob in enumerate(probs):
            by = y + i * (bar_height + 4)
            c = COLOR_MAP.get(i, (200, 200, 200))
            # 背景
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_width, by + bar_height), (50, 50, 50), -1)
            # 填充
            fill_w = int(prob * bar_width)
            cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + bar_height), c, -1)
            # 标签
            short_label = EMOTION_LABELS[i][:3]
            cv2.putText(frame, short_label, (bar_x + bar_width + 4, by + bar_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # FPS
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def process_frame(self, frame):
        """
        处理单帧图像：检测 → 预处理 → 预测 → 平滑 → 绘制

        Returns:
            frame: 标注后的图像
        """
        # 计时
        start = time.time()

        # 镜像翻转
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = self.face_detector.detect(gray, frame)

        # 逐人脸处理
        for i, face_rect in enumerate(faces):
            preprocessed = self.preprocess_face(gray, face_rect)
            if preprocessed is None:
                continue

            probs = self.predict(preprocessed)
            smoothed = self.smooth_predict(i, probs)
            emotion_idx = np.argmax(smoothed)

            fps = self._compute_fps()
            self.draw_ui(frame, face_rect, emotion_idx, smoothed, fps)

        self.frame_times.append(time.time() - start)
        return frame

    def _compute_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def process_image(self, image_path, output_path=None):
        """
        处理单张图片。

        Parameters:
            image_path: 输入图片路径
            output_path: 输出图片路径（可选）
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[识别] 无法读取图片: {image_path}")
            return

        result = self.process_frame(frame)

        # 打印结果
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detect(gray, frame)
        for i, face_rect in enumerate(faces):
            preprocessed = self.preprocess_face(gray, face_rect)
            if preprocessed is None:
                continue
            probs = self.predict(preprocessed)
            smoothed = self.smooth_predict(i, probs)
            idx = np.argmax(smoothed)
            print(f"  人脸 {i+1}: {EMOTION_LABELS[idx]} ({smoothed[idx]:.1%})")

        if output_path:
            cv2.imwrite(output_path, result)
            print(f"[识别] 结果已保存: {output_path}")
        else:
            cv2.imshow("Emotion Recognition", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def run_camera(recognizer):
    """
    实时摄像头识别主循环。

    快捷键：
        Q - 退出
        S - 截图保存
        H - 显示/隐藏帮助面板
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[识别] 无法打开摄像头")
        return

    print("[识别] 摄像头已打开，按 Q 退出，S 截图，H 帮助")
    show_help = True
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = recognizer.process_frame(frame)

        # 帮助面板
        if show_help:
            help_text = [
                "Q - Quit",
                "S - Screenshot",
                "H - Toggle Help",
            ]
            for j, line in enumerate(help_text):
                cv2.putText(result, line, (10, result.shape[0] - 60 + j * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Emotion Recognition", result)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.png"
            cv2.imwrite(filename, result)
            print(f"[识别] 截图已保存: {filename}")
        elif key == ord("h"):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()
