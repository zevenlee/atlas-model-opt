#根据输出结果json文件，对检测框准确度作出评估
# 保存为 evaluate_coco.py
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# 配置路径 (需替换为实际路径)
gt_annotation_path = "instances_val2017.json"  # COCO真实标注文件
dt_result_path = "correct_predictions.json"  # 你的预测结果文件

# 加载标注和预测
coco_gt = COCO(gt_annotation_path)
coco_dt = coco_gt.loadRes(dt_result_path)

# 初始化评估器 (使用bbox指标)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# 执行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 获取详细指标 (可选)
print("\n详细指标：")
stats = coco_eval.stats
metrics = {
    "AP": stats[0],        # mAP@[0.5:0.95]
    "AP50": stats[1],      # mAP@0.5
    "AP75": stats[2],      # mAP@0.75
    "AP_small": stats[3],  # 小目标AP
    "AP_medium": stats[4], # 中目标AP
    "AP_large": stats[5],  # 大目标AP
    "AR_max1": stats[6],   # 最大检测数1时的AR
    "AR_max10": stats[7],  # 最大检测数10时的AR
    "AR_100": stats[8]     # 最大检测数100时的AR
}
print(json.dumps(metrics, indent=2))
