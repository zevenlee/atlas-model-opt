#!/bin/bash

set -e
atc --model=./resnet50.onnx --framework=5 --output=./resnet50 --soc_version=Ascend310B4 --input_shape="input:1,3,224,224" --enable_small_channel=1

# --model :待转换模型路径，输入模型
# --framework： 输入模型原始框架类型
# --input_shaep： 模型输入数据shape
# -- soc_version： 模型转化时制定的芯片版本
# --output ：输出模型名
#更多参数见官方手册
