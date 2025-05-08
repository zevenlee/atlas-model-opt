#!/bin/bash

set -e
read -p "请输入模型输入大小(320/640/1280)： " resolution
atc --model=./yolov5.onnx --framework=5 --output=./yolov5 --soc_version=Ascend310B4 --input_shape="images:1,3,$resolution,$resolution" 


# --model :待转换模型路径，输入模型
# --framework： 输入模型原始框架类型
# --input_shaep： 模型输入数据shape
# -- soc_version： 模型转化时制定的芯片版本
# --output ：输出模型名
#更多参数见官方手册
