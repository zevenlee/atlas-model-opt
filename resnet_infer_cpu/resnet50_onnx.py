import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime 
from torchvision import transforms
import resource
import torchvision.transforms as transforms

IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']

class ONNXClassifier:
    """ONNX模型推理封装类"""
    def __init__(self, model_path, label_list):
        """
        初始化ONNX分类器
        :param model_path: ONNX模型文件路径
        :param label_list: 分类标签列表
        """
        # 初始化ONNX推理会话
        self.session = onnxruntime.InferenceSession(model_path)
        self.label_list = label_list
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"成功加载ONNX模型，输入形状：{self.input_shape}")

    def run(self, input_data):
        """
        执行推理
        :param input_data: 预处理后的输入数据（numpy数组）
        :return: (top1索引, 前五预测结果字典)
        """
        # 执行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # 解析输出结果
        preds = np.squeeze(outputs[0])
        top5_idx = np.argsort(preds)[::-1][:5]
        
        # 转换为结果字典
        pred_dict = {self.label_list[i]: float(preds[i]) for i in top5_idx}
        return top5_idx[0], pred_dict

def pre(image_path):
    """PyTorch风格预处理（保持不变）"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).numpy()

def save_image(path, pred_dict, output_path):
    """保存结果图片（保持不变）"""
    font = ImageFont.truetype('font.ttf', 20)
    color = "#fff"
    im = Image.open(path).resize((800, 500))
    draw = ImageDraw.Draw(im)
    
    start_y = 20
    for label, pred in pred_dict.items():
        draw.text((20, start_y), f'{label}: {pred:.2f}', font=font, fill=color)
        start_y += 30
    
    im.save(os.path.join(output_path, os.path.basename(path)))

def load_label_mappings(synset_path):
    """加载标签映射（保持不变）"""
    folder_to_labels = {}
    with open(synset_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(' ', 1)
            if len(parts) < 2: continue
            folder, labels = parts[0], [l.strip().lower() for l in parts[1].split(',')]
            folder_to_labels[folder] = labels
    return folder_to_labels


    def main():
# 参数配置（修改模型路径）
    model_path = "./resnet50_quant_labelfree.onnx"  # ONNX模型路径
    dataset_path = "./classify/data/ILSVRC2012_val"
    output_path = "./output_onnx"
    synset_path = "./classify/label/synset_words.txt"
    class_path = "./classify/label/imagenet_classes.txt"

    # 初始化配置
    os.makedirs(output_path, exist_ok=True)
    with open(class_path) as f:
        idx2label = [line.strip() for line in f]
    # 加载标签映射
    folder_to_labels = load_label_mappings(synset_path)
    
    # 初始化ONNX分类器
    classifier = ONNXClassifier(model_path, idx2label)
    # 初始化图片数量参数
    max_images = 256    
    # 遍历数据集
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[1] in IMG_EXT:
                image_paths.append(os.path.join(root, file))
    
    # 性能统计
    total, correct = 0, 0
    total_start = time.time()
    inference_time = 0

    with open(os.path.join(output_path, "results.txt"), "w") as f:
        processed_count = 0 
        for img_path in image_paths:
            if processed_count >= max_images:
                break
            # 获取真实标签
            folder_name = os.path.basename(os.path.dirname(img_path))
            true_labels = folder_to_labels.get(folder_name, [])

            # 预处理
            img = pre(img_path)
            
            # 推理处理
            infer_start = time.time()
            pred_index, pred_dict = classifier.run(img)
            inference_time += time.time() - infer_start

            # 结果验证
            pred_label = idx2label[pred_index].lower()
            is_correct = any(tl.lower() in pred_label or 
                            pred_label in tl.lower() 
                            for tl in true_labels)
            processed_count += 1
            correct += is_correct
            total = processed_count

            # 保存结果
            #save_image(img_path, pred_dict, output_path)
            f.write(f"{img_path}\t{pred_label}\t{true_labels}\t{is_correct}\n")

    # 最终统计
    total_time = time.time() - total_start
    accuracy = correct / total if total else 0
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    stats = f"""\n
    ========== CPU性能统计 ==========
    处理图片总数: {total}
    推理耗时: {inference_time:.2f}s
    总运行时间: {total_time:.2f}s
    CPU峰值内存占用: {peak_mem:.2f} MB
    分类准确率: {accuracy:.4%}
    ================================
    """
    print(stats)
    with open(os.path.join(output_path, "stats.txt"), "w") as f:
        f.write(stats)

if __name__ == '__main__':
    main()
