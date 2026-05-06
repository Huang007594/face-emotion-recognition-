"""
训练模块：数据加载 → 增强 → 编译 → 训练 → 评估 → 保存曲线
支持文件夹结构和 CSV 格式
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data import download_fer2013, load_fer2013, get_data_augmentation
from model import build_advanced_cnn

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical


def train_model(args):
    """
    完整训练流程。
    args: 命令行参数对象，包含 data_path, epochs, batch_size, save_dir 等字段
    """
    # ---- 1. 数据 ----
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = download_fer2013(args.save_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(data_path)

    num_classes = 7
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # ---- 2. 数据增强 ----
    datagen = get_data_augmentation()
    datagen.fit(X_train)

    # ---- 3. 构建模型 ----
    model = build_advanced_cnn(input_shape=(48, 48, 1), num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- 4. 回调 ----
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "best_model.h5")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ---- 5. 训练 ----
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
    )

    # ---- 6. 评估 ----
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n[训练] 测试集 Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # ---- 7. 绘制训练曲线 ----
    plot_history(history, args.save_dir)

    # ---- 8. 保存最终模型 ----
    final_path = os.path.join(args.save_dir, "final_model.h5")
    model.save(final_path)
    print(f"[训练] 最终模型已保存: {final_path}")

    return model, history


def plot_history(history, save_dir):
    """绘制并保存训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[训练] 训练曲线已保存: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FER2013 训练脚本")
    parser.add_argument("--data_path", type=str, default=None,
                        help="数据集路径（文件夹或 CSV，不指定则自动下载）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--save_dir", type=str, default="saved_model", help="模型保存目录")
    args = parser.parse_args()

    train_model(args)
