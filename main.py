"""
应用入口：命令行接口，调度 train / camera / image 子命令
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="人脸表情识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py train                        # 使用默认参数训练
  python main.py train --data_path ./data     # 指定数据集路径训练
  python main.py camera --model best_model.h5 # 实时摄像头识别
  python main.py image --model best_model.h5 --input photo.jpg  # 单张图片识别
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data_path", type=str, default=None,
                              help="数据集路径（文件夹或 CSV，不指定则自动下载）")
    train_parser.add_argument("--epochs", type=int, default=100,
                              help="训练轮数（默认 100）")
    train_parser.add_argument("--batch_size", type=int, default=64,
                              help="批大小（默认 64）")
    train_parser.add_argument("--save_dir", type=str, default="saved_model",
                              help="模型保存目录（默认 saved_model）")

    # ---- camera ----
    camera_parser = subparsers.add_parser("camera", help="实时摄像头识别")
    camera_parser.add_argument("--model", type=str, default="saved_model/best_model.h5",
                               help="模型文件路径")
    camera_parser.add_argument("--smooth_window", type=int, default=10,
                               help="滑动窗口长度（默认 10）")
    camera_parser.add_argument("--confidence", type=float, default=0.5,
                               help="人脸检测置信度阈值（默认 0.5）")

    # ---- image ----
    image_parser = subparsers.add_parser("image", help="单张图片识别")
    image_parser.add_argument("--model", type=str, default="saved_model/best_model.h5",
                              help="模型文件路径")
    image_parser.add_argument("--input", type=str, required=True,
                              help="输入图片路径")
    image_parser.add_argument("--output", type=str, default=None,
                              help="输出图片路径（可选）")
    image_parser.add_argument("--smooth_window", type=int, default=10,
                              help="滑动窗口长度（默认 10）")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # ---- 调度 ----
    if args.command == "train":
        from train import train_model
        train_model(args)

    elif args.command == "camera":
        from recognizer import EmotionRecognizer, run_camera
        recognizer = EmotionRecognizer(
            model_path=args.model,
            smooth_window=args.smooth_window,
            confidence_threshold=args.confidence,
        )
        run_camera(recognizer)

    elif args.command == "image":
        from recognizer import EmotionRecognizer
        recognizer = EmotionRecognizer(
            model_path=args.model,
            smooth_window=args.smooth_window,
        )
        recognizer.process_image(args.input, args.output)


if __name__ == "__main__":
    main()
