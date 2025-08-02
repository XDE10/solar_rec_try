import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
from src.models.yolo.yolov5.models.experimental import attempt_load
from src.models.yolo.yolov5.utils.general import non_max_suppression, scale_coords
from src.models.yolo.yolov5.utils.plots import plot_one_box
from src.models.yolo.yolov5.utils.torch_utils import select_device
from src.models.yolo.yolov5.utils.augmentations import letterbox

def main():
    # 设置路径
    input_dir = r"D:\Users\Mr.Z\Desktop\mask"
    output_dir = r"D:\Users\Mr.Z\Desktop\yolov5x_results"
    metrics_dir = r"D:\Users\Mr.Z\Desktop\yolov5x_metrics"
    model_path = r"D:\Solar-Panel\models\yolov5_X.pt"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 加载YOLOv5模型
    try:
        model = attempt_load(model_path, map_location='cpu')
        print("使用备选方法加载YOLOv5_X模型成功")
    except Exception as e:
        print(f"所有加载方法都失败: {e}")
        return
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 检测结果存储
    results_data = []
    
    # 处理每张图片
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(input_dir, image_file)
        
        # 检测
        results = model(image_path)
        
        # 保存检测结果图像
        results.save(save_dir=output_dir)
        
        # 提取检测结果数据
        if len(results.pandas().xyxy[0]) > 0:
            for _, row in results.pandas().xyxy[0].iterrows():
                results_data.append({
                    'image_name': image_file,
                    'class': row['class'],
                    'name': row['name'],
                    'confidence': round(row['confidence'], 4),
                    'x1': round(row['xmin'], 1),
                    'y1': round(row['ymin'], 1),
                    'x2': round(row['xmax'], 1),
                    'y2': round(row['ymax'], 1),
                    'width': round(row['xmax'] - row['xmin'], 1),
                    'height': round(row['ymax'] - row['ymin'], 1),
                    'area': round((row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin']), 1),
                })
    
    # 保存检测结果到CSV
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(metrics_dir, 'detection_results.csv'), index=False)
        
        # 计算统计信息
        stats_data = {
            'total_images': len(image_files),
            'images_with_detections': len(set(results_df['image_name'])),
            'total_detections': len(results_df),
            'avg_detections_per_image': round(len(results_df) / len(image_files), 2),
            'avg_confidence': round(results_df['confidence'].mean(), 4),
            'avg_object_area': round(results_df['area'].mean(), 1)
        }
        
        stats_df = pd.DataFrame([stats_data])
        stats_df.to_csv(os.path.join(metrics_dir, 'detection_stats.csv'), index=False)
        
        print(f"检测结果已保存至: {os.path.join(metrics_dir, 'detection_results.csv')}")
        print(f"检测统计已保存至: {os.path.join(metrics_dir, 'detection_stats.csv')}")
    
    print(f"批量检测完成，结果图像已保存至: {output_dir}")

if __name__ == "__main__":
    main()