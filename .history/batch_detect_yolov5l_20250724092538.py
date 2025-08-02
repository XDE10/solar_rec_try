import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 导入本地YOLOv5代码
import sys
sys.path.append('src/models/yolo/yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.augmentations import letterbox

def main():
    # 设置路径
    input_dir = r"D:\Users\Mr.Z\Desktop\mask"
    output_dir = r"D:\Users\Mr.Z\Desktop\yolov5l_results"
    metrics_dir = r"D:\Users\Mr.Z\Desktop\yolov5l_metrics"
    model_path = r"D:\Solar-Panel\models\yolov5_L.pt"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 加载YOLOv5模型
    device = select_device('')  # 空字符串表示自动选择设备
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max())
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    
    print("YOLOv5_L模型加载成功")
    
    # 获取类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 检测结果存储
    results_data = []
    
    # 处理每张图片
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(input_dir, image_file)
        
        # 读取图像
        img0 = cv2.imread(image_path)  # BGR
        if img0 is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 预处理图像
        img = letterbox(img0, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred = model(img)[0]
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        # 处理检测结果
        det = pred[0]
        if len(det):
            # 调整坐标到原始图像大小
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # 处理每个检测框
            im0 = img0.copy()
            for *xyxy, conf, cls in det:
                # 保存检测结果数据
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                
                results_data.append({
                    'image_name': image_file,
                    'class': int(cls),
                    'name': names[int(cls)],
                    'confidence': round(float(conf), 4),
                    'x1': round(float(x1), 1),
                    'y1': round(float(y1), 1),
                    'x2': round(float(x2), 1),
                    'y2': round(float(y2), 1),
                    'width': round(float(x2 - x1), 1),
                    'height': round(float(y2 - y1), 1),
                    'area': round(float((x2 - x1) * (y2 - y1)), 1),
                })
                
                # 在图像上绘制检测框
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(0, 255, 0), line_thickness=2)
                
            # 保存结果图像
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, im0)
    
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