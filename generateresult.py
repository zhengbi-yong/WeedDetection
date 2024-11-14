import csv
import os
from ultralytics import YOLO

if __name__ == "__main__":
    # 加载 YOLO 模型
    model = YOLO("./runs/detect/train5/weights/best.pt")
    
    # 进行预测，设置置信度和 IoU 阈值
    results = model.predict(
        source="./weed-detection/test/images",  # 测试图像的文件夹
        show=False,
        conf=0.20,  # 置信度阈值，根据需要调整
        iou=0.40    # IoU 阈值，根据需要调整
    )

    # 打开一个 CSV 文件以写入结果
    with open("prediction_results.csv", "w", newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入 CSV 文件的头部
        csvwriter.writerow(["ID", "image_id", "class_id", "x_min", "y_min", "width", "height"])

        id_counter = 1  # 用于生成唯一的ID
        for result in results:
            image_path = result.path
            image_id = os.path.splitext(os.path.basename(image_path))[0]  # 提取图像ID

            if result.boxes is not None:
                for box in result.boxes:
                    # 提取边界框信息
                    xyxy = box.xyxy[0]  # 边界框坐标 (x_min, y_min, x_max, y_max)
                    class_id = int(box.cls[0])  # 类别ID
                    x_min, y_min, x_max, y_max = map(int, xyxy)  # 将坐标转换为整数

                    # 计算宽度和高度
                    width = x_max - x_min
                    height = y_max - y_min

                    # 写入一行数据
                    csvwriter.writerow([id_counter, image_id, class_id, x_min, y_min, width, height])
                    id_counter += 1

        # 如果行数不足 4999，补足格式为 [ID, 99999, 9, 0, 0, 0, 0] 的行
        while id_counter <= 4999:
            csvwriter.writerow([id_counter, 99999, 9, 0, 0, 0, 0])
            id_counter += 1
