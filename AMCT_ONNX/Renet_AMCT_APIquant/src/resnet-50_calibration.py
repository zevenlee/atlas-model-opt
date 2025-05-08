from PIL import Image, ImageDraw, ImageFont  # 图片处理库，用于在图片上画出推理结果

import os
import argparse
from torchvision import transforms
import torchvision.transforms as transforms
import cv2
import numpy as np
import onnxruntime as ort

import amct_onnx as amct

PATH = os.path.realpath('./')
IMG_DIR = os.path.join(PATH, 'data/images')
LABEL_FILE = os.path.join(IMG_DIR, 'image_label.txt')

PARSER = argparse.ArgumentParser(description='amct_onnx resnet-50 quantization sample.')
PARSER.add_argument('--nuq', dest='nuq', action='store_true', help='whether use nuq')
ARGS = PARSER.parse_args()
#判断是否有NUQ权重文件（选择均匀/非均匀量化）
if ARGS.nuq:
    OUTPUTS = os.path.join(PATH, 'outputs/nuq')
else:
    OUTPUTS = os.path.join(PATH, 'outputs/calibration')

TMP = os.path.join(OUTPUTS, 'tmp')

#读取imagenet数据集标签与图片
def get_labels_from_txt(label_file):
    """Read all images' name and label from label_file"""
    images = []
    labels = []
    with open(label_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            images.append(line.split(' ')[0])
            labels.append(int(line.split(' ')[1]))
    return images, labels

def pre(image_path):
    """pytorch图片预处理"""
    transform = transforms.Compose([
        transforms.Resize(256),  # 将图像大小调整为 256x256
        transforms.CenterCrop(224),  # 中心裁剪为 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 预处理并添加批量维度
    return image.numpy()

#输入图片预处理
def prepare_image_input(
    images, height=256, width=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Read image files to blobs [batch_size, 3, 224, 224]"""
    #input_tensor = np.zeros((1, 3, crop_size, crop_size), np.float32)

    imgs = np.zeros((1, 3, height, width), np.float32)
    im_data = cv2.imread(images)
    im_data = cv2.resize(im_data, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)

    imgs[:, :, :] = im_data.transpose(2, 0, 1).astype(np.float32)

    h_off = int((height - crop_size) / 2)
    w_off = int((width - crop_size) / 2)
    input_tensor = imgs[:, :, h_off: (h_off + crop_size), w_off: (w_off + crop_size)]
    # trans uint8 image data to float
    input_tensor /= 255
    # do channel-wise reduce mean value
    for channel in range(input_tensor.shape[1]):
        input_tensor[:, channel, :, :] -= mean[channel]
    # do channel-wise divide std
    for channel in range(input_tensor.shape[1]):
        input_tensor[:, channel, :, :] /= std[channel]

    return input_tensor

#图片后处理，按top5的置信度排序
def img_postprocess(probs, labels):
    """Do image post-process"""
    # calculate top1 and top5 accuracy
    top1_get = 0
    top5_get = 0
    prob_size = probs.shape[1]
    for index, label in enumerate(labels):
        top5_record = (probs[index, :].argsort())[prob_size - 5: prob_size]
        if label == top5_record[-1]:
            top1_get += 1
            top5_get += 1
        elif label in top5_record:
            top5_get += 1
    return float(top1_get) / len(labels), float(top5_get) / len(labels)

#onnx推理函数 前向传播
def onnx_forward(onnx_model, batch_size=1, iterations=160):
    """forward"""
    ort_session = ort.InferenceSession(onnx_model, amct.AMCT_SO)

    images, labels = get_labels_from_txt(LABEL_FILE)
    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = pre(images[i])

        output = ort_session.run(None, {'input': input_batch})
        top1, top5 = img_postprocess(output[0], labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
        print('****************iteration:{}*****************'.format(i))
        print('top1_acc:{}'.format(top1))
        print('top5_acc:{}'.format(top5))
    print('******top1:{}'.format(top1_total / iterations))
    print('******top5:{}'.format(top5_total / iterations))
    return top1_total / iterations, top5_total / iterations


def main():
    """main"""
    #配置参数
    #原始模型推理
    model_file = './model/resnet50.onnx'
    print('[INFO] Do original model test:')
    ori_top1, ori_top5 = onnx_forward(model_file, 1, 2)
	
    config_json_file = os.path.join(TMP, 'config.json')
    skip_layers = []
    batch_num = 1
    #均匀/非均匀量化
    if ARGS.nuq:
        amct.create_quant_config(
            config_file=config_json_file, model_file=model_file, skip_layers=skip_layers, batch_num=batch_num,
            activation_offset=True, config_defination='./src/nuq_conf/nuq_quant.cfg')
    else:
        amct.create_quant_config(
            config_file=config_json_file, model_file=model_file, skip_layers=skip_layers, batch_num=batch_num,
            activation_offset=True, config_defination=None)

    # do conv+bn fusion, weights calibration and generate
    #         calibration model
    scale_offset_record_file = os.path.join(TMP, 'record.txt')
    modified_model = os.path.join(TMP, 'modified_model.onnx')
    amct.quantize_model(
        config_file=config_json_file, model_file=model_file, modified_onnx_file=modified_model,
        record_file=scale_offset_record_file)
    onnx_forward(modified_model, 1, batch_num)

    # save final model, one for onnx do fake quant test, one
    #         deploy model for ATC
    result_path = os.path.join(OUTPUTS, 'resnet-101')
    amct.save_model(modified_model, scale_offset_record_file, result_path)

    # run fake_quant model test
    #保证量化结果正确
    print('[INFO] Do quantized model test:')
    quant_top1, quant_top5 = onnx_forward('%s_%s' % (result_path, 'fake_quant_model.onnx'), 1, 5)
    print('[INFO] ResNet101 before quantize top1:{:>10} top5:{:>10}'.format(ori_top1, ori_top5))
    print('[INFO] ResNet101 after quantize  top1:{:>10} top5:{:>10}'.format(quant_top1, quant_top5))


if __name__ == '__main__':
    main()
