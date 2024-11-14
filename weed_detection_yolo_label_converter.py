import os
import json

def circle_to_bbox(shape, img_width, img_height, class_id):
    # 提取圆心和半径
    x_center, y_center = shape['points'][0]
    x_radius, y_radius = shape['points'][1]
    radius = ((x_radius - x_center) ** 2 + (y_radius - y_center) ** 2) ** 0.5

    # 计算矩形框的左上角和右下角
    x_min = max(0, x_center - radius)
    y_min = max(0, y_center - radius)
    x_max = min(img_width, x_center + radius)
    y_max = min(img_height, y_center + radius)

    # 转换为YOLO格式：相对坐标
    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height
    width_normalized = (x_max - x_min) / img_width
    height_normalized = (y_max - y_min) / img_height

    return f"{class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}"

def convert_annotations_to_yolo(input_dir, output_dir):
    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入文件夹中的所有json文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name.replace('.json', '.txt'))

            # 打开并解析json文件
            with open(input_file_path, 'r') as json_file:
                data = json.load(json_file)

            img_width, img_height = data['imageWidth'], data['imageHeight']
            shapes = data['shapes']
            
            yolo_labels = []
            for shape in shapes:
                # 根据标签的内容设置 class_id
                if shape['label'] == 'mq':
                    class_id = 1
                else:
                    class_id = 0

                if shape['shape_type'] == 'circle':
                    yolo_label = circle_to_bbox(shape, img_width, img_height, class_id)
                    yolo_labels.append(yolo_label)
            
            # 将YOLO格式的标签写入到新的txt文件中
            with open(output_file_path, 'w') as yolo_file:
                for label in yolo_labels:
                    yolo_file.write(label + '\n')

    print(f"转换完成！所有标签已保存到 {output_dir}")

# 示例用法：
input_folder = "./weed-detection/train/labels_origin"  # 输入文件夹路径，包含json格式的标签
output_folder = "./weed-detection/train/labels"  # 输出文件夹路径，保存转换后的YOLO标签
convert_annotations_to_yolo(input_folder, output_folder)
