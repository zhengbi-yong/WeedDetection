import os

# 临时解决 OMP 错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

if __name__ == "__main__":
    # 使用预训练模型
    model = YOLO("yolo11x")  # 可根据需要选择不同大小的模型，如 yolov8n.pt, yolov8s.pt, yolov8m.pt 等
    
    # 训练模型，支持新的训练过程
    results = model.train(
        data="weed-detection/WeedDetection.yaml",  # 数据集配置文件
        epochs=1000,  # 增加训练轮数
        # resume="path/to/last.pt",  # 如果有有效的检查点，取消注释并设置路径
        batch=16,  # 批量大小，根据GPU显存调整
        imgsz=640,  # 输入图片大小
        optimizer='Adam',  # 优化器
        lr0=0.001,  # 初始学习率
        weight_decay=0.0005,  # 权重衰减
        device=1,  # 使用的GPU设备
        workers=4,  # 数据加载的工作线程数
        verbose=True,  # 启用详细输出
        # 数据增强参数（根据需要调整）
        hsv_h=0.015,  # 色调增强范围
        hsv_s=0.7,    # 饱和度增强范围
        hsv_v=0.4,    # 亮度增强范围
        degrees=0.0,  # 旋转角度
        translate=0.1,  # 平移比例
        scale=0.5,     # 缩放比例
        shear=0.0,     # 剪切角度
        perspective=0.0,  # 透视变换
        flipud=0.0,    # 上下翻转概率
        fliplr=0.5,    # 左右翻转概率
        mosaic=1.0,    # 马赛克增强概率
        mixup=0.0      # 混合增强概率
    )
