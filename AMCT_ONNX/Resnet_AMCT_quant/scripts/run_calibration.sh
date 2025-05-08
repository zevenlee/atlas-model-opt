#!/bin/bash

set -e
#step1：将data/images目录下的jpg文件处理为data/calibration目录下的bin文件。
python3 ./src/process_data.py
#step2：进行模型量化操作。
#amct_onnx calibration --model model/resnet101_v11.onnx --input_shape "input:16,3,224,224" --data_type "float32" --data_dir ./data/calibration/ --save_path ./results/resnet101_v1
amct_onnx calibration --model ./model/resnet50.onnx --save_path ./result/resnet50_sq --input_shape="input:1,3,224,224" --data_dir "./data/calibration" --data_types "float32"
