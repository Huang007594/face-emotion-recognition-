"""
数据模块：FER2013 数据集的下载、解析与预处理
支持文件夹结构（kagglehub msambare/fer2013）和 CSV 格式
"""

import os
import numpy as np

# 统一标签顺序（与文件夹名和 CSV 索引对齐）
EMOTION_LABELS = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

# 文件夹名 → 标签索引
FOLDER_TO_INDEX = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
    "sad": 4, "surprise": 5, "neutral": 6,
}


def download_fer2013(save_dir: str = "data") -> str:
    """
    自动下载 FER2013 数据集（通过 kagglehub）。
    返回数据集根目录路径（包含 train/ 和 test/ 子目录）。
    """
    import kagglehub

    os.makedirs(save_dir, exist_ok=True)
    print("[数据] 正在通过 kagglehub 下载数据集...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print(f"[数据] 数据集已下载到: {path}")

    # 检查是否为文件夹结构
    if os.path.isdir(os.path.join(path, "train")):
        return path

    # 检查是否为 CSV 格式
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)

    raise FileNotFoundError("数据集格式不支持，期望文件夹结构或 CSV 文件。")


def load_fer2013_from_folder(data_dir: str):
    """
    从文件夹结构加载 FER2013 数据集。
    期望目录结构：data_dir/train/{angry,disgust,...} 和 data_dir/test/{...}

    返回 (X_train, y_train), (X_val, y_val), (X_test, y_test)
    像素值归一化到 [0, 1]，形状为 (N, 48, 48, 1)
    """
    import cv2

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # 加载训练集
    for folder_name, label_idx in FOLDER_TO_INDEX.items():
        folder = os.path.join(train_dir, folder_name)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        total = len(files)
        val_count = max(1, int(total * 0.1))  # 10% 作为验证集

        for i, fname in enumerate(files):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = img.astype("float32") / 255.0
            img = img.reshape(48, 48, 1)

            if i < val_count:
                X_val.append(img)
                y_val.append(label_idx)
            else:
                X_train.append(img)
                y_train.append(label_idx)

    # 加载测试集
    if os.path.isdir(test_dir):
        for folder_name, label_idx in FOLDER_TO_INDEX.items():
            folder = os.path.join(test_dir, folder_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (48, 48))
                img = img.astype("float32") / 255.0
                img = img.reshape(48, 48, 1)
                X_test.append(img)
                y_test.append(label_idx)

    # 打乱训练集和验证集
    X_train, y_train = _shuffle(X_train, y_train)
    X_val, y_val = _shuffle(X_val, y_val)
    X_test, y_test = _shuffle(X_test, y_test)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"[数据] 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    print(f"[数据] 标签分布(训练): {np.bincount(y_train.astype(int))}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_fer2013_from_csv(csv_path: str):
    """
    从 CSV 格式加载 FER2013 数据集（兼容旧版 fer2013.csv）。
    """
    import pandas as pd

    print(f"[数据] 正在解析 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for _, row in df.iterrows():
        emotion = int(row["emotion"])
        pixels = np.array(row["pixels"].split(), dtype="float32")
        pixels = pixels.reshape(48, 48, 1) / 255.0
        usage = row["Usage"]

        if usage == "Training":
            X_train.append(pixels)
            y_train.append(emotion)
        elif usage == "PublicTest":
            X_val.append(pixels)
            y_val.append(emotion)
        else:
            X_test.append(pixels)
            y_test.append(emotion)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"[数据] 训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_fer2013(data_path: str):
    """
    自动判断数据格式并加载。
    data_path: 数据集目录（含 train/ 文件夹）或 CSV 文件路径
    """
    if data_path.endswith(".csv"):
        return load_fer2013_from_csv(data_path)
    else:
        return load_fer2013_from_folder(data_path)


def _shuffle(X, y):
    """打乱数据"""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return [X[i] for i in indices], [y[i] for i in indices]


def get_data_augmentation():
    """
    返回一个配置好的 ImageDataGenerator，用于数据增强。
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest",
    )


if __name__ == "__main__":
    # 快速测试：下载并解析数据
    data_path = download_fer2013()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(data_path)
    print(f"标签分布(训练): {np.bincount(y_train.astype(int))}")
