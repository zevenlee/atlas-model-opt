#修正COCO数据集标准不正确的地方，以满足COCO官方的评估工具要求

import json
import re
# 输入输出路径配置
coco_ann_path = "instances_val2017.json"  # COCO标准标注文件
pred_path = "predictions_om.json"        # 需要修正的预测文件
output_path = "fixed_predictions.json"   # 修正后输出路径

def unify_label_name(name):
    name = re.sub(r'[^a-zA-Z0-9]','',name).strip()
    return re.sub(r'\s+','',name)

# 加载数据
with open(coco_ann_path, 'r') as f:
    coco_data = json.load(f)
with open(pred_path, 'r') as f:
    pred_data = json.load(f)

# 构建双向映射字典
coco_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
coco_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# 步骤1：建立旧ID到新ID的映射关系
id_mapping = {}
for pred_cat in pred_data["categories"]:
    original_mane = pred_cat["name"]
    pred_cat["name"] = unify_label_name(original_mane)
    # 通过名称匹配COCO标准ID
    if pred_cat["name"] in coco_name_to_id :
        coco_name_to_id = {unify_label_name(cat["name"]): cat["id"] for cat in coco_data["categories"]}
        new_id = coco_name_to_id[pred_cat["name"]]
        id_mapping[pred_cat["id"]] = new_id
    else:
        # 名称不在COCO标准中的特殊处理
        raise ValueError(f"发现未知类别：{pred_cat['name']}，请检查类别命名")

# 步骤2：构建完全对齐COCO标准的categories
fixed_categories = coco_data["categories"]  # 直接使用COCO的类别定义

# 步骤3：更新所有annotations中的category_id
fixed_annotations = []
for ann in pred_data["annotations"]:
    original_id = ann["category_id"]
    if original_id in id_mapping:
        new_ann = ann.copy()
        new_ann["category_id"] = id_mapping[original_id]
        fixed_annotations.append(new_ann)
    else:
        print(f"丢弃无效类别ID {original_id} 的标注")

# 构建最终结果 (保持COCO标准结构)
fixed_data = {
    "info": pred_data.get("info", {}),
    "images": pred_data["images"],
    "annotations": fixed_annotations,
    "categories": fixed_categories  # 使用COCO标准类别列表
}

# 保存结果
with open(output_path, 'w') as f:
    json.dump(fixed_data, f, indent=2)

# 验证结果
print("修正完成，关键验证：")
print(f"- 类别总数：{len(fixed_categories)} (应与COCO的80一致)")
print(f"- 首类别验证：ID={fixed_categories[0]['id']} name={fixed_categories[0]['name']}")
print(f"- 最后标注验证：原ID=13 → 新ID={id_mapping.get(13, 'N/A')}")
