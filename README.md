# 人脸表情识别系统

基于深度学习的人脸表情识别项目，支持摄像头实时识别和单张图片识别，可识别 7 种表情。

## 识别表情

| 索引 | 表情 | 英文 | 颜色 |
|:----:|------|------|------|
| 0 | 😠 愤怒 | Angry | 红 |
| 1 | 🤮 厌恶 | Disgust | 橙 |
| 2 | 😨 恐惧 | Fear | 紫 |
| 3 | 😄 开心 | Happy | 绿 |
| 4 | 😢 悲伤 | Sad | 蓝 |
| 5 | 😲 惊讶 | Surprise | 黄 |
| 6 | 😐 中性 | Neutral | 灰 |

## 项目结构

```
├── main.py              # 入口：命令行调度（train / camera / image）
├── model.py             # 模型：残差 + 深度可分离卷积 + SE 注意力的 CNN
├── data.py              # 数据：FER2013 下载、加载、增强
├── detector.py          # 检测：人脸检测（DNN / Haar Cascade，兼容中文路径）
├── recognizer.py        # 识别：预处理 + 预测 + 滑动窗口平滑 + UI 绘制
├── train.py             # 训练：完整训练流程 + 曲线绘制
├── requirements.txt     # 依赖
├── run_train.bat        # 一键训练
├── run_camera.bat       # 一键摄像头识别
├── run_image.bat        # 一键图片识别
├── models/              # 人脸检测模型（可选手动放置）
├── data/                # FER2013 数据集（首次训练自动下载）
└── saved_model/         # 训练保存的模型和曲线
    ├── best_model.h5          # 验证准确率最高的模型
    ├── final_model.h5         # 最终 epoch 的模型
    └── training_curves.png    # 训练曲线图
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

双击 `run_train.bat`，或命令行：

```bash
python main.py train --epochs 50 --batch_size 64
```

首次运行会自动从 Kaggle 下载 FER2013 数据集。训练完成后模型保存在 `saved_model/` 目录。

训练参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | 自动下载 | 数据集路径（文件夹或 CSV） |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--save_dir` | saved_model | 模型保存目录 |

### 3. 摄像头实时识别

双击 `run_camera.bat`，或命令行：

```bash
python main.py camera --model saved_model/best_model.h5
```

快捷键：

| 按键 | 功能 |
|------|------|
| Q | 退出 |
| S | 截图保存 |
| H | 显示/隐藏帮助 |

### 4. 单张图片识别

双击 `run_image.bat` 并输入图片路径，或命令行：

```bash
python main.py image --model saved_model/best_model.h5 --input 照片.jpg
```

可选参数 `--output result.jpg` 保存标注后的图片。

## 模型架构

采用轻量级 CNN 设计，核心组件：

- **残差连接（Residual Connection）**：缓解梯度消失，支持更深网络
- **深度可分离卷积（SeparableConv2D）**：大幅减少参数量
- **SE 注意力模块（Squeeze-and-Excitation）**：自适应通道注意力

```
Input (48×48×1)
  → Conv2D(32) + BN + ReLU + MaxPool
  → ResBlock(64) ×2 + MaxPool
  → ResBlock(128) ×2 + MaxPool
  → ResBlock(256) ×2
  → GlobalAvgPool + Dropout(0.5)
  → Dense(7, softmax)
```

## 训练策略

| 策略 | 配置 |
|------|------|
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=5) |
| 早停 | EarlyStopping (patience=20, 恢复最佳权重) |
| 模型检查点 | 保存 val_accuracy 最高的模型 |
| 数据增强 | 旋转 ±15°、平移 ±15%、水平翻转、缩放 ±15% |

## 训练结果

在 FER2013 数据集上训练 50 个 Epoch：

| 指标 | 数值 |
|------|------|
| 测试集准确率 | 65.97% |
| 最佳验证准确率 | 66.74%（Epoch 40） |
| 最终训练准确率 | 71.93% |

> FER2013 数据集上 66% 左右的准确率属于正常水平，该数据集存在标注噪声和类别不平衡问题。

## 依赖

```
tensorflow>=2.10.0
opencv-python>=4.6.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scipy>=1.10.0
kagglehub>=0.1.0
```

## 数据集说明

本项目使用 [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) 数据集，首次训练时自动下载。

如果自动下载失败，可手动下载：

1. 访问 https://www.kaggle.com/datasets/msambare/fer2013 下载数据集
2. 解压到项目目录下的 `data/` 文件夹，结构如下：
   ```
   data/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```
3. 训练时指定数据路径：`python main.py train --data_path ./data`

## 注意事项

- **中文路径**：项目已兼容中文路径，OpenCV 模型加载通过字节流方式绕过编码问题
- **GPU 支持**：Windows 原生 TensorFlow >= 2.11 不支持 GPU，使用 CPU 训练（如需 GPU 请使用 WSL2）
- **Kaggle 下载**：首次训练需要 Kaggle 账号认证，按提示配置 `kaggle.json`
