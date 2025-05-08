#!/bin/bash
set -e

amct_onnx convert --model model/mobilenetv2_qat.onnx  --save_path output/result
