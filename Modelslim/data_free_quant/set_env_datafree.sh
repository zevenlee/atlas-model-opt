#!/bin/bash

set -e
#调用squant_ptq接口，Data-free量化模式环境配置

pip3 install numpy		
#大于等于1.21.6且小于等于1.23.0

pip3 install onnx		
#大于等于1.14.0	

pip3 install onnxruntime	
#大于等于1.14.1

pip3 install torch==1.11.0	
#CPU版本torch，1.8.1/1.11.0

pip3 install onnx-simplifier	
#大于等于0.3.10


