import os
import pandas as pd
import cv2  # 需要 opencv-python, 使用 "pip install opencv-python" 安装
import argparse
from tqdm import tqdm

def find_image_path(source_img_dir, base_name):
    """
    在源图片目录中查找具有相同基本名称的图片文件。
    支持常见的图片格式。
    """
    for ext in ['.jpg', '.jpeg', '.png']:
        potential_path = os.path.join(source_img_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None

def generate_metrics_report(results_dir, source_img_dir):
    """
    分析YOLOv5检测结果，生成详细的CSV报告和统计摘要。

    Args:
        results_dir (str): 包含所有模型结果文件夹的根目录。
                           (例如 'D:/.../detection_results')
        source_img_dir (str): 包含原始输入图像的目录。
                              (例如 'D:/.../mask')
    """
    print(f"开始处理结果目录: {results_dir}")
    print(f"原始图片目录: {source_img_dir}\n")

    # 简单类别映射，您可以根据您的 `data.yaml` 文件进行扩展
    class_map = {
        0: 'solar_panel',
        # 1: 'другой_класс',  # 如果有更多类别，可以添加
    }

    # 遍历 results_dir 中的每一个子文件夹 (如 'yolov5l_results')
    for model_folder in os.listdir(results_dir):
        model_result_path = os.path.join(results_dir, model_folder)
        
        if not os.path.isdir(model_result_path):
            continue

        labels_dir = os.path.join(model_result_path, 'labels')
        
        # 检查 'labels' 文件夹是否存在
        if not os.path.isdir(labels_dir):
            print(f"在 '{model_result_path}' 中未找到 'labels' 文件夹，跳过。")
            continue

        print(f"--- 正在处理模型: {model_folder} ---")

        # 创建 metrics 文件夹
        metrics_dir = os.path.join(model_result_path, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        results_data = []
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        if not label_files:
            print("未找到任何标签文件。")
            continue

        # 使用tqdm创建进度条
        for label_file in tqdm(label_files, desc=f"分析 {model_folder} 的标签"):
            base_name = os.path.splitext(label_file)[0]
            
            # 找到对应的原始图片以获取尺寸
            image_path = find_image_path(source_img_dir, base_name)
            if not image_path:
                print(f"警告: 找不到 '{base_name}' 对应的原始图片，跳过此标签文件。")
                continue
            
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape

            # 读取标签文件中的每一行
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    # 解析YOLO格式: class_id x_center_norm y_center_norm w_norm h_norm [conf]
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    confidence = float(parts[5]) if len(parts) > 5 else 0.0

                    # 将归一化坐标转换为像素坐标
                    w_px = w_norm * img_width
                    h_px = h_norm * img_height
                    x_center_px = x_center_norm * img_width
                    y_center_px = y_center_norm * img_height

                    # 计算 x1, y1, x2, y2
                    x1 = x_center_px - (w_px / 2)
                    y1 = y_center_px - (h_px / 2)
                    x2 = x_center_px + (w_px / 2)
                    y2 = y_center_px + (h_px / 2)

                    area = w_px * h_px

                    results_data.append({
                        'image_name': os.path.basename(image_path),
                        'class_id': class_id,
                        'class_name': class_map.get(class_id, 'unknown'),
                        'confidence': round(confidence, 4),
                        'x1': round(x1, 1),
                        'y1': round(y1, 1),
                        'x2': round(x2, 1),
                        'y2': round(y2, 1),
                        'width': round(w_px, 1),
                        'height': round(h_px, 1),
                        'area': round(area, 1),
                    })

        # --- 保存统计结果 ---
        if not results_data:
            print("未从标签文件中收集到任何数据。")
            continue

        # 1. 保存详细的检测结果
        results_df = pd.DataFrame(results_data)
        csv_results_path = os.path.join(metrics_dir, 'detection_results.csv')
        results_df.to_csv(csv_results_path, index=False)
        print(f"\n详细检测结果已保存至: {csv_results_path}")

        # 2. 计算并保存统计摘要
        total_source_images = len([f for f in os.listdir(source_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        stats_data = {
            'total_source_images': total_source_images,
            'images_with_detections': len(results_df['image_name'].unique()),
            'total_detections': len(results_df),
            'avg_detections_per_image': round(len(results_df) / total_source_images, 2) if total_source_images > 0 else 0,
            'avg_confidence': round(results_df['confidence'].mean(), 4) if not results_df['confidence'].empty else 0,
            'avg_object_area': round(results_df['area'].mean(), 1) if not results_df['area'].empty else 0
        }
        
        stats_df = pd.DataFrame([stats_data])
        csv_stats_path = os.path.join(metrics_dir, 'detection_stats.csv')
        stats_df.to_csv(csv_stats_path, index=False)
        print(f"检测统计摘要已保存至: {csv_stats_path}\n")

if __name__ == '__main__':
    # --- 配置区域 ---
    # argparse 允许我们从命令行方便地传入参数
    parser = argparse.ArgumentParser(description="从YOLOv5检测结果生成CSV报告。")
    
    # 修改这里的 default 值，使其指向您的目录
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default=r'D:\Users\Mr.Z\Desktop\detection_results',
        help='包含所有模型结果文件夹的根目录。'
    )
    parser.add_argument(
        '--source-img-dir', 
        type=str, 
        default=r'D:\Users\Mr.Z\Desktop\mask',
        help='包含原始输入图像的目录。'
    )
    
    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.isdir(args.results_dir):
        print(f"错误: 结果目录不存在 -> '{args.results_dir}'")
    elif not os.path.isdir(args.source_img_dir):
        print(f"错误: 源图片目录不存在 -> '{args.source_img_dir}'")
    else:
        generate_metrics_report(args.results_dir, args.source_img_dir)