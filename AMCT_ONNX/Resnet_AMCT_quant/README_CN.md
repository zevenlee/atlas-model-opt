# ONNX 框架命令行示例

## 1. ONNX ResNet-101 分类网络模型量化

### 1.1 量化前提

+ **模型准备**  
下载模型文件到 [model](./model/) 目录。

+ **数据集准备**  
推理过程中需要使用和模型相匹配的数据集。请测试图片存入 “images” 文件夹并将文件夹放到 [data](./data/) 目录下。

+ **校准集准备**  
校准集用来产生量化因子，保证精度。校准集与数据集相同。

### 1.2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录，其中 images 文件夹内部包含有 160 张用于校准和测试的图片：

+ [data](./data/)
  + images
+ [model](./model/)
  + resnet101_v11.onnx

请在当前目录执行如下命令运行示例程序：
```none
bash ./scripts/run_calibration.sh
```
执行成功会在当前目录生成results文件夹，文件夹下有resnet101_v11_deploy_model.onnx和resnet101_v11_fake_quant_model.onnx两个文件

### 1.3 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_onnx.log ，并在当前目录下生成以下内容：

+ results: 存放量化后模型的文件夹。
  + resnet101_v11_deploy_model.onnx: 量化部署模型，即量化后的可在昇腾 AI 处理器部署的模型文件。
  + resnet101_v11_fake_quant_model.onnx: 量化仿真模型，即量化后的可在 ONNX 执行框架 ONNX Runtime 进行精度仿真的模型文件。
  + resnet101_v11_quant.json：融合信息文件。


