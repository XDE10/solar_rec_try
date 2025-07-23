import json
import cv2
import numpy as np
import os
import sys

# --- 这里是你的文件路径 ---
# 使用 r"..." 来确保 Windows 路径被正确读取
json_path = r"D:\必应下载\labels_my-project-name_2025-07-22-02-01-29.json"
image_path = r"D:\Users\Mr.Z\Desktop\照片及掩码\2.png"
# -------------------------

print("开始处理...")
print(f"JSON 文件: {json_path}")
print(f"目标图片: {image_path}")

# 确保图片文件存在
if not os.path.exists(image_path):
    print(f"错误：找不到图片文件！请检查路径：{image_path}")
    sys.exit(1)

# --- 设置输出掩码的文件名和路径 ---
# 获取原图片的目录和文件名（不含扩展名）
output_directory = os.path.dirname(image_path)
base_filename = os.path.splitext(os.path.basename(image_path))[0]
# 确保文件名不含乱码
output_mask_path = os.path.join(output_directory, f"{base_filename}_mask.png")

print(f"掩码将保存至: {output_mask_path}")
# -------------------------

# 读取 JSON 文件
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：找不到 JSON 文件！请检查路径：{json_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"错误：JSON 文件格式不正确！")
    sys.exit(1)

# 在 JSON 数据中找到与你提供的图片匹配的信息
target_image_filename = os.path.basename(image_path)
image_info = None
for img in data.get('images', []):
    if img['file_name'] == target_image_filename:
        image_info = img
        break

if not image_info:
    print(f"错误：在JSON文件中没有找到关于图片 '{target_image_filename}' 的记录。")
    sys.exit(1)

# 获取图片尺寸和ID
image_id = image_info['id']
height = image_info['height']
width = image_info['width']
print(f"在JSON中找到图片信息：ID={image_id}, 尺寸={width}x{height}")

# 创建一个和原图一样大小的纯黑色画布
mask = np.zeros((height, width), dtype=np.uint8)

# 筛选出属于这张图片的所有标注
annotations_for_image = [ann for ann in data.get('annotations', []) if ann['image_id'] == image_id]

if not annotations_for_image:
    print("警告：找到了图片信息，但在JSON中没有找到任何对应的标注。将生成一个纯黑的掩码。")
else:
    print(f"找到了 {len(annotations_for_image)} 个标注，正在绘制多边形...")
    # 将所有多边形（用白色）画到黑色画布上
    for ann in annotations_for_image:
        for seg in ann.get('segmentation', []):
            try:
                poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], color=(255))
            except Exception as e:
                print(f"警告：绘制多边形时出错：{e}")

# 保存最终生成的掩码图片
try:
    success = cv2.imwrite(output_mask_path, mask)
    if success:
        print("\n--- 成功！---")
        print(f"掩码图片已生成并保存至:")
        print(output_mask_path)
        # 检查文件是否真的被创建
        if os.path.exists(output_mask_path):
            print(f"文件大小: {os.path.getsize(output_mask_path)} 字节")
        else:
            print("警告：文件似乎未被创建！")
    else:
        print("\n--- 失败！---")
        print("imwrite返回失败。可能的原因：路径无效、权限问题或磁盘空间不足。")
except Exception as e:
    print(f"\n--- 错误！---")
    print(f"保存文件时出错：{e}")
    print("请检查输出路径是否有写入权限。") 