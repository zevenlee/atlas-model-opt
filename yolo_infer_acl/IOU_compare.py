import json
import numpy as np

def calculate_iou(box1, box2):
    # 计算两个检测框的IoU（交并比）
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

def analyze_quant_effect(original_json, quant_json, iou_threshold=0.5):
    with open(original_json, 'r') as f:
        orig_data = {item['image']: item for item in json.load(f)}
    with open(quant_json, 'r') as f:
        quant_data = {item['image']: item for item in json.load(f)}

    total_matched = 0
    total_orig_boxes = 0
    total_quant_boxes = 0
    class_consistency = 0

    for img_name in orig_data:
        if img_name not in quant_data:
            continue
        orig_boxes = orig_data[img_name]['boxes']
        quant_boxes = quant_data[img_name]['boxes']
        orig_classes = orig_data[img_name]['classes']
        quant_classes = quant_data[img_name]['classes']
        total_orig_boxes += len(orig_boxes)
        total_quant_boxes += len(quant_boxes)

        # 匹配检测框
        matched_pairs = []
        for i, orig_box in enumerate(orig_boxes):
            for j, quant_box in enumerate(quant_boxes):
                iou = calculate_iou(orig_box, quant_box)
                if iou >= iou_threshold and orig_classes[i] == quant_classes[j]:
                    matched_pairs.append((i, j))
                    class_consistency += 1
        total_matched += len(matched_pairs)

    # 计算指标
    match_rate_orig = total_matched / total_orig_boxes if total_orig_boxes > 0 else 0
    match_rate_quant = total_matched / total_quant_boxes if total_quant_boxes > 0 else 0
    class_consistency_rate = class_consistency / total_matched if total_matched > 0 else 0

    #print(f"原始模型检测框匹配率（IoU≥{iou_threshold}）: {match_rate_orig:.2%}")
    print("label-free量化：")
    print(f"量化模型检测框匹配率（IoU≥{iou_threshold}）: {match_rate_quant:.2%}")
    print(f"类别一致性（匹配框中类别相同的比例）: {class_consistency_rate:.2%}")

def compare_confidence(original_json, quant_json):
    with open(original_json, 'r') as f:
        orig_data = {item['image']: item for item in json.load(f)}
    with open(quant_json, 'r') as f:
        quant_data = {item['image']: item for item in json.load(f)}

    conf_diffs = []
    for img_name in orig_data:
        if img_name not in quant_data:
            continue
        orig_confs = orig_data[img_name]['confidences']
        quant_confs = quant_data[img_name]['confidences']
        # 假设检测框已匹配（需先调用匹配函数）
        for orig_conf, quant_conf in zip(orig_confs, quant_confs):
            conf_diffs.append(orig_conf - quant_conf)

    avg_diff = np.mean(conf_diffs)
    print(f"平均置信度差值（原始模型 - 量化模型）: {avg_diff:.4f}")

# 调用函数
analyze_quant_effect('yolov5m_om.json', 'yolov5m_quant_labelfree.json', iou_threshold=0.5)
compare_confidence('yolov5m_om.json', 'yolov5m_quant_labelfree.json')
