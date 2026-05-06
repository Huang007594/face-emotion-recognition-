"""
模型模块：带残差连接 + 深度可分离卷积 + SE 注意力的 CNN
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def squeeze_excitation_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation 注意力模块。
    """
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    return layers.Multiply()([input_tensor, se])


def residual_block(x, filters, kernel_size=3, stride=1):
    """
    可复用的残差块：深度可分离卷积 + BN + SE + 跳跃连接。
    """
    shortcut = x

    # 如果通道数或空间尺寸变化，需要对 shortcut 做投影
    if stride > 1 or x.shape[-1] != filters:
        shortcut = layers.SeparableConv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 主路径
    y = layers.SeparableConv2D(
        filters, kernel_size, strides=stride, padding="same", use_bias=False
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.SeparableConv2D(
        filters, kernel_size, strides=1, padding="same", use_bias=False
    )(y)
    y = layers.BatchNormalization()(y)

    # SE 注意力
    y = squeeze_excitation_block(y)

    # 跳跃连接
    y = layers.Add()([shortcut, y])
    y = layers.ReLU()(y)
    return y


def build_advanced_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    组装完整网络：Stem → 3 个残差块组 → GAP → FC → Softmax
    """
    inputs = layers.Input(shape=input_shape)

    # Stem：普通卷积做特征提取
    x = layers.Conv2D(32, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)  # 48→24

    # 残差块组 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)  # 24→12

    # 残差块组 2
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)  # 12→6

    # 残差块组 3
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # 全局平均池化 + 分类头
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, x, name="AdvancedCNN_FER")
    model.summary()
    return model


if __name__ == "__main__":
    model = build_advanced_cnn()
    print(f"\n模型总参数量: {model.count_params():,}")
