import json 
import os
import numpy as np  # 用于对多维数组进行计算
from PIL import Image, ImageDraw, ImageFont  # 图片处理库，用于在图片上画出推理结果
import time
import sys
import resource
import acl  # acl推理相关接口
from torchvision import transforms
import torchvision.transforms as transforms
import torch
#npu使用率监测(线程回显抓取)
import subprocess
import re
import time
from threading import Thread, Event

class NpuMonitor:
    def __init__(self, device_id=0, interval=0.05):
        self.device_id = device_id      # NPU设备ID
        self.interval = interval        # 轮询间隔（秒）
        self.max_usage = 0              # 记录峰值使用率
        self._monitor_event = Event()   # 控制监控线程启停

    def _get_aicore_usage(self):
        """执行npu-smi命令并解析AICore使用率"""
        try:
            # 执行命令并获取输出
            cmd = f"npu-smi info -t usages -i {self.device_id} | grep \"Aicore Usage Rate\""
            result = subprocess.run(cmd, shell=True, capture_output=True,text=True,timeout=1)
            output = result.stdout
              
            # 正则匹配使用率（示例输出行：'AICore Usage Rate : 25%'）
            match = re.search(r"AiCore Usage Rate\s*\(\%\)\s*:\s*(\d+)", output,re.IGNORECASE)
            return int(match.group(1)) if match else 0
        except Exception as e:
            print(f"[Error] 获取NPU使用率失败: {e}")
            return 0

    def _monitor_loop(self):
        """监控循环：持续记录峰值使用率"""
        while not self._monitor_event.is_set():
            current_usage = self._get_aicore_usage()
            if current_usage > self.max_usage:
                self.max_usage = current_usage
            time.sleep(self.interval)

    def start(self):
        """启动监控线程"""
        self._monitor_event.clear()
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop(self):
        """停止监控线程并返回峰值使用率"""
        self._monitor_event.set()
        self.monitor_thread.join()
        return self.max_usage



ACL_MEM_MALLOC_HUGE_FIRST = 0  # 内存分配策略
ACL_SUCCESS = 0  # 成功状态值
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']  # 所支持的图片格式

class Net(object):
    def __init__(self, device_id, model_path, idx2label_list):
        self.device_id = device_id  # 设备id
        self.model_path = model_path  # 模型路径
        self.model_id = None  # 模型id
        self.context = None  # 用于管理资源，
        self.model_desc = None  # 模型描述信息，包括模型输入个数、输入维度、输出个数、输出维度等信息
        self.load_input_dataset = None  # 输入数据集，aclmdlDataset类型
        self.load_output_dataset = None  # 输出数据集，aclmdlDataset类型

        self.init_resource()  # 初始化 acl 资源
        self.idx2label_list = idx2label_list  # 加载的标签列表

    def init_resource(self):
        """初始化 acl 相关资源"""
        print("init resource stage:")

        ret = acl.init()  # 初始化 acl
        check_ret("acl.init", ret)

        ret = acl.rt.set_device(self.device_id)  # 指定 device
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)  # 创建 context
        check_ret("acl.rt.create_context", ret)

        self.model_id, ret = acl.mdl.load_from_file(self.model_path)  # 加载模型
        check_ret("acl.mdl.load_from_file", ret)

        self.model_desc = acl.mdl.create_desc()  # 创建描述模型基本信息的数据类型
        print("init resource success")

        ret = acl.mdl.get_desc(self.model_desc, self.model_id)  # 根据模型ID获取模型基本信息
        check_ret("acl.mdl.get_desc", ret)


    def _gen_input_dataset(self, input_list):
        ''' 组织输入数据的dataset结构 '''
        input_num = acl.mdl.get_num_inputs(self.model_desc)  # 根据模型信息得到模型输入个数
        self.load_input_dataset = acl.mdl.create_dataset()  # 创建输入dataset结构
        for i in range(input_num):
            item = input_list[i]  # 获取第 i 个输入数据
            data = acl.util.bytes_to_ptr(item.tobytes())  # 获取输入数据字节流
            size = item.size * item.itemsize  # 获取输入数据字节数
            dataset_buffer = acl.create_data_buffer(data, size)  # 创建输入dataset buffer结构, 填入输入数据
            _, ret = acl.mdl.add_dataset_buffer(self.load_input_dataset, dataset_buffer)  # 将dataset buffer加入dataset
        #print("create model input dataset success")


    def _gen_output_dataset(self):
        ''' 组织输出数据的dataset结构 '''
        output_num = acl.mdl.get_num_outputs(self.model_desc)  # 根据模型信息得到模型输出个数
        self.load_output_dataset = acl.mdl.create_dataset()  # 创建输出dataset结构
        for i in range(output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)  # 获取模型输出个数
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)  # 为每个输出申请device内存
            dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)  # 创建输出的data buffer结构,将申请的内存填入data buffer
            _, ret = acl.mdl.add_dataset_buffer(self.load_output_dataset, dataset_buffer)  # 将 data buffer 加入输出dataset
        #print("create model output dataset success")

    def run(self, images):
        self._gen_input_dataset(images)
        self._gen_output_dataset()
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        result = []
        for i in range(acl.mdl.get_num_outputs(self.model_desc)):
            buffer = acl.mdl.get_dataset_buffer(self.load_output_dataset, i)
            data = acl.get_data_buffer_addr(buffer)
            size = acl.get_data_buffer_size(buffer)
            narray = acl.util.ptr_to_bytes(data, size)
            dims = acl.mdl.get_cur_output_dims(self.model_desc, i)[0]['dims']
            result.append(np.frombuffer(narray, dtype=np.float32).reshape(dims))
        pred_index, pred_dict = self._parse_result(result)
        self._destroy_dataset()
        return pred_index, pred_dict

    def _parse_result(self, result):
        vals = np.array(result).flatten()
        top_k = vals.argsort()[::-1]
        pred_dict = {self.idx2label_list[j]: vals[j] for j in top_k[:5]}
        return top_k[0], pred_dict


    def _print_result(self, result):
        """打印预测结果"""
        vals = np.array(result).flatten()  # 将结果展开为一维
        top_k = vals.argsort()[-1:-6:-1]  # 将置信度从大到小排列，并得到top5的下标

        #print("======== top5 inference results: =============")
        pred_dict = {}
        for j in top_k:
            #print(f'{self.idx2label_list[j]}: {vals[j]}')  # 打印出对应类别及概率
            pred_dict[self.idx2label_list[j]] = vals[j]  # 将类别信息和概率存入 pred_dict
        return pred_dict

    def _destroy_dataset(self):
        """ 释放模型输入输出数据 """
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)  # 获取输入buffer个数
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)  # 获取每个输入buffer
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)  # 销毁每个输入buffer (销毁 aclDataBuffer 类型)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)  # 销毁输入数据 (销毁 aclmdlDataset类型的数据)
            check_ret("acl.mdl.destroy_dataset", ret)


    def release_resource(self):
        """释放 acl 相关资源"""
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)  # 卸载模型
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)  # 释放模型描述信息
            self.model_desc = None

        if self.context:
            ret = acl.rt.destroy_context(self.context)  # 释放 Context
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)  # 释放 device 资源
        check_ret("acl.rt.reset_device", ret)

        ret = acl.finalize()  # ACL去初始化
        check_ret("acl.finalize", ret)
        print('Resources released successfully.')

def check_ret(message, ret):
    ''' 用于检查各个返回值是否正常，若否，则抛出对应异常信息 '''
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}".format(message, ret))


def preprocess_img(input_path):
    """caffe图片预处理"""
    # 循环加载图片
    input_path = os.path.abspath(input_path)  # 得到当前图片的绝对路径
    with Image.open(input_path) as image_file:
        image_file = image_file.resize((256, 256),Image.BICUBIC)  # 缩放图片
        img = np.array(image_file)  # 转为numpy数组

    # 获取图片的高和宽
    height = img.shape[0]
    width = img.shape[1]

    # 对图片进行切分，取中间区域
    h_off = (height - 224) // 2
    w_off = (width - 224) // 2
    crop_img = img[h_off:h_off+224, w_off:w_off+224, :]

    # 转换bgr格式、数据类型、颜色空间、数据维度等信息
    img = crop_img[:, :, ::-1].astype(np.float32)
    img -= np.array([103.94, 116.78, 123.68])
    img = img.transpose(2,0,1)[np.newaxis, ...]
    
    return img

def pre(image_path):
    """pytroch图片预处理"""
    transform = transforms.Compose([
        transforms.Resize(256),  # 将图像大小调整为 256x256
        transforms.CenterCrop(224),  # 中心裁剪为 224x224
        transforms.ToTensor(),  # 转换为张量
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  #预处理并添加批量维度
    #print("形状",image.shape)

    image = image.permute(0,2,3,1) #将NHWC转NCHW
    image_uint8 = (image * 255 ).type(torch.uint8)
    image = image_uint8
    #print("形状",image.shape)
    #print("数据类型",image.dtype)
    #print("数据范围:Min ={:.4f},Max={:.4f}".format(image.min(),image.max()))

    image = image.numpy()
    return image
    



def save_image(path, pred_dict, output_path):
    """保存预测图片"""
    font = ImageFont.truetype('font.ttf', 20)  # 指定字体和字号
    color = "#fff"  # 指定颜色
    im = Image.open(path)  # 打开图片
    im = im.resize((800, 500))  # 对图片进行缩放

    start_y = 20  # 在图片上画分类结果时的纵坐标初始值
    draw = ImageDraw.Draw(im)  # 准备画图
    for label, pred in pred_dict.items():
        draw.text(xy = (20, start_y), text=f'{label}: {pred:.2f}', font=font, fill=color)  # 将预测的类别与置信度添加到图片
        start_y += 30  # 每一行文字往下移30个像素

    im.save(os.path.join(output_path, os.path.basename(path)))  # 保存图片到输出路径

def load_label_mappings(synset_path):
    """标签映射加载"""
    folder_to_labels = {}
    with open(synset_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(' ', 1)
            if len(parts) < 2:
                print(f"Warning: Invalid line format: {line}")
                continue
            folder, labels = parts[0], [l.strip().lower() for l in parts[1].split(',')]
            folder_to_labels[folder] = labels
    return folder_to_labels

def main():
    device = 0
    model_path = "./resnet50_aipp_ch.om"
    dataset_path = "./classify/data/ILSVRC2012_val"
    output_path = "./output_om_aipp"
    synset_path = "./classify/label/synset_words.txt"
    class_path = "./classify/label/imagenet_classes.txt"

    # 初始化配置
    os.makedirs(output_path, exist_ok=True)
    with open(class_path) as f:
        idx2label = [line.strip() for line in f]
    folder_to_labels = load_label_mappings(synset_path)

    # 遍历数据集
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[1] in IMG_EXT:
                image_paths.append(os.path.join(root, file))
    # 初始化图片数量参数
    max_images = 1      
    # 初始化模型
    net = Net(device, model_path, idx2label)
    total, correct = 0, 0
    total_start = time.time()
    inference_time = 0
    # 初始化监控器
    monitor = NpuMonitor(device_id=0, interval=0.05)
    # 启动监控
    monitor.start()
    with open(os.path.join(output_path, "results.txt"), "w") as f:
        processed_count = 0 
        for img_path in image_paths:
            if processed_count >= max_images:
                break
            # 获取真实标签
            folder_name = os.path.basename(os.path.dirname(img_path))
            true_labels = folder_to_labels.get(folder_name, [])

            # 关键修改：使用PyTorch预处理
            img = pre(img_path)  # 替换原有预处理
            
            # 推理处理
            infer_start = time.time()
            pred_index, pred_dict = net.run([img])
            inference_time += time.time() - infer_start

            # 结果验证（增强匹配逻辑）
            pred_label = idx2label[pred_index].lower()
            is_correct = any(
                tl.lower() in pred_label or  # 允许部分匹配
                pred_label in tl.lower() 
                for tl in true_labels
            )
            processed_count += 1
            correct += is_correct
            total = processed_count

            # 保存详细结果
            f.write(f"{img_path}\t{pred_label}\t{true_labels}\t{is_correct}\n")


    peak_usage = monitor.stop()
    peak_npu_mem = 3513 *0.01* peak_usage
    # 性能统计
    total_time = time.time() - total_start
    accuracy = correct / total if total else 0
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # 转换为MB

    # 输出统计信息
    stats = f"""\n
    ========== Performance Statistics ==========
    图片数量: {total}
    推理耗时: {inference_time:.2f}s
    总运行时间: {total_time:.2f}s
    CPU峰值内存占用: {peak_mem:.2f} MB
    NPU峰值内存占用: {peak_npu_mem:.2f} MB
    准确率: {accuracy:.4%}
    =============================================
    """
    print(stats)
    with open(os.path.join(output_path, "stats.txt"), "w") as f:
        f.write(stats)

    net.release_resource()

if __name__ == '__main__':
    main()
