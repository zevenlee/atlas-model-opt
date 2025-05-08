
#将输出的预测文件转换成coco评估支持的json格式
import json

# 加载您当前的预测文件
with open("fixed_predictions.json") as f:
    data = json.load(f)

# 提取annotations并转换格式
coco_results = []
for ann in data["annotations"]:
    # 确保字段名和类型正确
    coco_results.append({
        "image_id": int(ann["image_id"]),  # 必须为整数
        "category_id": int(ann["category_id"]),  # COCO类别ID从1开始
        "bbox": [float(x) for x in ann["bbox"]],  # 必须为float数组
        "score": float(ann["score"])  # 置信度转为float
    })

# 保存为正确格式
with open("correct_predictions.json", "w") as f:
    json.dump(coco_results, f, indent=2)


