#!/bin/bash

set -e

atc --framework=5 --soc_version=$Ascend310B4
--model= resnet50.onnx
--insert_op_conf=resnet50_aipp.cfg 
--output=resnet50_aipp

#- framework：原始网络模型框架类型，3表示TensorFlow框架。

#- soc_version：指定模型转换时昇腾AI处理器的版本，例如Ascend310。

#- model：原始网络模型文件路径，含文件名。

#- insert_op_conf：AIPP预处理配置文件路径，含文件名。

#- output：转换后的*.om模型文件路径，含文件名，转换成功后，文件名自动以.om后缀结尾。
