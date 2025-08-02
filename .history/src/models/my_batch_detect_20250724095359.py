import os
import sys
import torch # 在您的本地环境中，这行代码可以正常工作

# 确保项目内部的模块可以被正确导入
# 这会把 'D:\Solar-Panel\src' 添加到Python的搜索路径中
# 请确保 D:\Solar-Panel\src 是您项目的src目录
# 如果您将脚本保存在 src/models/ 目录下，可能不需要下面这两行
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.insert(0, project_root)

# 这是一个关键的导入，它使用了项目内部的yolov5检测脚本
from yolo.yolov5 import detect as yolo_eval


def yolo_testing(**kwargs):
    """调用内置的YOLOv5检测函数"""
    params = get_yolo_params(**kwargs)
    print(f"--- 使用以下参数运行检测 ---\n{params}\n--------------------------")
    yolo_eval.run(**params)


def get_yolo_params(name, source, project='results', model='best.pt', device='cuda:0'):
    """构建传递给YOLOv5的参数字典"""
    return {
        'source': source,
        'weights': model,
        'imgsz': [256, 256],  # 图像尺寸
        'save_txt': True,       # 保存txt格式的标签文件
        'save_conf': True,      # 在txt标签中保存置信度
        'project': project,     # 结果保存的根目录
        'name': name,           # 结果保存的子目录名
        'line_thickness': 2,    # 边界框线条粗细
        'hide_labels': False,   # 是否隐藏标签
        'hide_conf': False,     # 是否隐藏置信度
        'device': device,
        'exist_ok': True,       # 允许覆盖已有的结果文件夹
    }


def main():
    # --- 1. 配置区域 ---
    # 在这里修改你的路径
    
    # 包含所有待检测图片的文件夹
    input_dir = r"D:\Users\Mr.Z\Desktop\mask" 
    
    # 所有输出的根目录
    base_output_dir = r"D:\Users\Mr.Z\Desktop\detection_results"
    
    # 定义要运行的模型列表
    models_to_run = [
        {
            "name": "yolov5l_results", # 定义结果文件夹的名称
            "path": r"D:\Solar-Panel\models\yolov5_L.pt" 
        },
        {
            "name": "yolov5x_results", # 定义结果文件夹的名称
            "path": r"D:\Solar-Panel\models\yolov5_X.pt" 
        }
    ]

    # 自动检测使用CPU还是GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用的设备: {device}")

    # --- 2. 处理流程 ---
    
    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    # 循环处理每个模型
    for model_config in models_to_run:
        model_name_for_folder = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\n{'='*20}")
        print(f"正在使用模型: {model_path}")
        print(f"{'='*20}")
        
        # 检查模型文件是否存在
        if not os.path.isfile(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在。跳过此模型。")
            continue

        # 调用核心检测函数
        yolo_testing(
            name=model_name_for_folder,
            source=input_dir,
            project=base_output_dir,
            model=model_path,
            device=device
        )
        
        print(f"模型 '{model_path}' 处理完成。")
        print(f"结果已保存至: {os.path.join(base_output_dir, model_name_for_folder)}")

if __name__ == '__main__':
    main()