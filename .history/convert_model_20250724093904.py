from ultralytics import YOLO

# --- 配置区域 ---
# 在这里填入你的旧模型路径和希望保存的新模型路径
models_to_convert = [
    {
        "old_path": r"D:\Solar-Panel\models\yolov5_L.pt",
        "new_path": r"D:\Solar-Panel\models\yolov5_L_new.pt"
    },
    {
        "old_path": r"D:\Solar-Panel\models\yolov5_X.pt",
        "new_path": r"D:\Solar-Panel\models\yolov5_X_new.pt"
    }
]
# --- 结束配置 ---

print("开始转换模型...")
for model_info in models_to_convert:
    old_path = model_info["old_path"]
    new_path = model_info["new_path"]
    try:
        print(f"正在加载旧模型: {old_path}")
        # 加载旧模型，此时会自动下载兼容包
        model = YOLO(old_path)
        
        # 以新的格式导出（本质是重新保存）
        # 我们导出为 PyTorch 格式，实际上就是 .pt
        model.export(format="pytorch", path=new_path)
        print(f"成功转换并保存新模型到: {new_path}\n")
    except Exception as e:
        print(f"转换模型 {old_path} 失败: {e}\n")

print("所有模型转换完成。")