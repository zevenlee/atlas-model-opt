#ONNX CPU YOLO模型推理，保存结果到json，并输出一张示例图片

import os
import json
import time
import random
import psutil
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

# ------------------------- 硬编码配置 -------------------------
MODEL_PATH = "yolov5s.onnx"  # 模型路径
IMAGES_FOLDER = "images"      # 图片文件夹
COCO_NAMES_PATH = "coco_names.txt"  # COCO类别标签文件
MODEL_INPUT_SIZE = 640        # 模型输入尺寸
CONF_THRESH = 0.3             # 置信度阈值
IOU_THRESH = 0.45             # NMS的IoU阈值
NUM_IMAGES = 200              # 处理图片数量

# ------------------------- 加载COCO类别 -------------------------
with open(COCO_NAMES_PATH, "r") as f:
    COCO_CLASSES = [line.strip() for line in f.readlines() if line.strip()]

# ------------------------- ONNX模型初始化 -------------------------
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# ------------------------- 核心功能函数 -------------------------
def preprocess_image(img_path):
    """图像预处理：缩放、归一化、转NCHW格式"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    img_np = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_np, axis=0)

def run_inference(input_data):
    """执行推理"""
    return session.run(["output0"], {"images": input_data})

def postprocess(outputs, orig_size):
    """后处理：解析检测框，应用NMS"""
    detections = outputs[0][0]  # [25200,85]
    boxes = []
    
    # 解析每个检测框
    for det in detections:
        cx, cy, w, h = det[:4]  # 中心坐标和宽高（绝对坐标，假设输入尺寸=模型尺寸）
        obj_conf = det[4].item()
        cls_probs = det[5:85]
        cls_id = np.argmax(cls_probs)
        total_conf = obj_conf * cls_probs[cls_id]
        
        if total_conf < CONF_THRESH:
            continue
        
        # 转换为中心点坐标 → [x_min, y_min, w, h]
        x_min = cx - w/2
        y_min = cy - h/2
        
        boxes.append({
            "bbox": [x_min, y_min, w, h],
            "score": total_conf,
            "category_id": cls_id + 1  # COCO类别从1开始
        })
    
    # NMS过滤
    boxes.sort(key=lambda x: x["score"], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [box for box in boxes if _iou(best["bbox"], box["bbox"]) < IOU_THRESH]
    return keep

def _iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return intersection / (area1 + area2 - intersection + 1e-9)

# ------------------------- 性能监控与输出 -------------------------
def monitor_memory():
    """获取当前进程内存占用（MB）"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def save_results(all_preds):
    """保存COCO格式预测结果"""
    with open("prediction_onnx.json", "w") as f:
        json.dump(all_preds, f, indent=2)

def visualize_example(img_path, pred_boxes):
    """可视化随机一张结果"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for box in pred_boxes:
        x, y, w, h = box["bbox"]
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
        label = f"{COCO_CLASSES[box['category_id']-1]} {box['score']:.2f}"
        draw.text((x, y-10), label, fill="red")
    img.save("result_onnx.jpg")

# ------------------------- 主程序 -------------------------
def main():
    # 获取图片列表
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:NUM_IMAGES]
    
    # 初始化性能统计
    start_time = time.time()
    initial_mem = monitor_memory()
    peak_mem = initial_mem
    all_preds = []
    
    # 处理每张图片
    for img_file in image_files:
        img_path = os.path.join(IMAGES_FOLDER, img_file)
        
        # 预处理
        input_data = preprocess_image(img_path)
        
        # 推理
        outputs = run_inference(input_data)
        
        # 后处理
        pred_boxes = postprocess(outputs, orig_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        
        # 记录结果
        all_preds.extend([{
            "image_id": os.path.splitext(img_file)[0],  # 假设文件名为COCO ID
            "category_id": box["category_id"],
            "bbox": [round(x, 2) for x in box["bbox"]],
            "score": round(box["score"], 3)
        } for box in pred_boxes])
        
        # 更新内存峰值
        current_mem = monitor_memory()
        peak_mem = max(peak_mem, current_mem)
    
    # 输出性能报告
    total_time = time.time() - start_time
    print(f"总耗时: {total_time:.2f}秒")
    print(f"内存峰值: {peak_mem - initial_mem:.2f} MB")
    print(f"平均内存占用: {(peak_mem + initial_mem) / 2:.2f} MB")
    
    # 保存结果
    save_results(all_preds)
    print("预测结果已保存至 prediction_onnx.json")
    
    # 可视化随机一张结果
    if all_preds:
        sample_img = random.choice(image_files)
        sample_preds = [p for p in all_preds if p["image_id"] == os.path.splitext(sample_img)[0]]
        visualize_example(os.path.join(IMAGES_FOLDER, sample_img), sample_preds)
        print("可视化结果已保存至 result_onnx.jpg")

if __name__ == "__main__":
    main()
