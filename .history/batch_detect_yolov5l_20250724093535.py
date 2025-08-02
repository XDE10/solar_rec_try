import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import cv2

def run_detection():
    # --- 1. 配置区域 ---
    # 在这里修改你的路径
    
    # 包含所有待检测图片的文件夹
    input_dir = r"D:\Users\Mr.Z\Desktop\mask" 
    
    # 所有输出的根目录
    base_output_dir = r"D:\Users\Mr.Z\Desktop\detection_results"
    
    # 定义要运行的模型列表
    # 你可以轻松地在这里添加或删除模型
    models_to_run = [
        {
            "name": "yolov5l",
            "path": r"D:\Solar-Panel\models\yolov5_L.pt" # YOLOv5L 模型文件路径
        },
        {
            "name": "yolov5x",
            "path": r"D:\Solar-Panel\models\yolov5_X.pt" # YOLOv5X 模型文件路径
        }
    ]

    # --- 2. 处理流程 ---
    
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    # 获取图像文件列表
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"警告: 在 '{input_dir}' 中没有找到任何图片文件。")
            return
    except FileNotFoundError:
        print(f"错误: 无法访问输入目录 '{input_dir}'。")
        return

    # 循环处理每个模型
    for model_config in models_to_run:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\n{'='*20}")
        print(f"正在使用模型: {model_name} ({model_path})")
        print(f"{'='*20}")

        # 检查模型文件是否存在
        if not os.path.isfile(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在。跳过此模型。")
            continue
            
        # 创建该模型的输出目录
        output_dir = os.path.join(base_output_dir, f"{model_name}_results")
        metrics_dir = os.path.join(base_output_dir, f"{model_name}_metrics")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # 加载YOLO模型
        try:
            model = YOLO(model_path)
            print(f"模型 '{model_name}' 加载成功。")
        except Exception as e:
            print(f"加载模型 '{model_name}' 失败: {e}")
            continue

        # 用于存储所有检测结果的列表
        results_data = []

        # 处理每张图片
        for image_file in tqdm(image_files, desc=f"使用 {model_name} 处理图像"):
            image_path = os.path.join(input_dir, image_file)
            
            try:
                # 使用YOLO模型进行预测
                # stream=True 模式更节省内存，适合处理大量图片
                results = model(image_path, verbose=False)
            except Exception as e:
                print(f"处理图片 '{image_file}' 时出错: {e}")
                continue

            # 从结果中提取信息
            for res in results:
                # 保存带标注的图片
                output_image_path = os.path.join(output_dir, image_file)
                res.save(filename=output_image_path)
                
                # 提取每个检测框的数据
                boxes = res.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    xyxy = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    results_data.append({
                        'image_name': image_file,
                        'class': class_id,
                        'name': class_name,
                        'confidence': round(confidence, 4),
                        'x1': round(x1, 1),
                        'y1': round(y1, 1),
                        'x2': round(x2, 1),
                        'y2': round(y2, 1),
                        'width': round(width, 1),
                        'height': round(height, 1),
                        'area': round(area, 1),
                    })

        # --- 3. 保存统计结果 ---
        if results_data:
            results_df = pd.DataFrame(results_data)
            csv_results_path = os.path.join(metrics_dir, 'detection_results.csv')
            results_df.to_csv(csv_results_path, index=False)
            
            # 计算统计信息
            if len(image_files) > 0:
                stats_data = {
                    'total_images': len(image_files),
                    'images_with_detections': len(set(results_df['image_name'])),
                    'total_detections': len(results_df),
                    'avg_detections_per_image': round(len(results_df) / len(image_files), 2) if image_files else 0,
                    'avg_confidence': round(results_df['confidence'].mean(), 4),
                    'avg_object_area': round(results_df['area'].mean(), 1)
                }
            else:
                stats_data = {} # No images, no stats
            
            stats_df = pd.DataFrame([stats_data])
            csv_stats_path = os.path.join(metrics_dir, 'detection_stats.csv')
            stats_df.to_csv(csv_stats_path, index=False)
            
            print(f"\n检测结果已保存至: {csv_results_path}")
            print(f"检测统计已保存至: {csv_stats_path}")
        else:
            print("\n未检测到任何目标。")

        print(f"模型 '{model_name}' 批量检测完成，结果图像已保存至: {output_dir}")

if __name__ == "__main__":
    run_detection()
