/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief AMCT custom ops
 *
 * @file dump_kernel.cpp
 *
 * @version 1.0
 */

#include "dump_kernel.h"
#include <sstream>
#include "amct_utils.h"
#include "util.h"


DUMPKernel::DUMPKernel(const OrtApi& api, const OrtKernelInfo* info)
    : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "batch_num", &bathNum_));
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");

    int64_t layerNum;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "layer_num", &layerNum));
    for (int i = 0; i < layerNum; ++i) {
        std::string attrName = "object_layer";
        attrName = attrName.append(std::to_string(i));
        std::string layerName = AmctUtils::GetStringAttr(api_, info, attrName);
        objectLayerNames_.push_back(layerName);
    }
    dumpDir_ = AmctUtils::GetStringAttr(api_, info, "dump_dir");
    dumpStamp_ = AmctUtils::GetStringAttr(api_, info, "dump_stamp");
}

void DUMPKernel::DumpData(const void* x, const int inputSize, const std::vector<int32_t>& inputShapeFlt,
    std::string objectLayerName)
{
    std::string trimedLayerName = AmctUtils::TrimTailSpace(objectLayerName);
    std::string trimedDumpDir = AmctUtils::TrimTailSpace(dumpDir_);
    std::string trimedDumpStamp = AmctUtils::TrimTailSpace(dumpStamp_);
    AmctUtils::ConvertLayerName(trimedLayerName, "/", "_");
    std::stringstream ss;
    ss << trimedDumpDir << '/' << trimedLayerName << \
        "_act_calibration_layer_" << trimedDumpStamp << "_" << std::to_string(currentBatch_) << ".bin";
    std::string fileName = ss.str();
    AmctUtils::AmctDumpData(fileName.c_str(), inputShapeFlt.data(), inputShapeFlt.size(), x, inputSize);
}

#if ORT_API_VERSION >= 16
OrtStatusPtr DUMPKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void DUMPKernel::Compute(OrtKernelContext* context)
{
    // dump count control
    currentBatch_++;
    if (currentBatch_ > bathNum_) {
        return;
    }
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);

    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
    // check input size
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);
    // get tensor shape info
    std::vector<int64_t> inputShape = AmctUtils::GetShape(api_, inputInfo);
    size_t shapeLen = inputShape.size() + 1;
    std::vector<int32_t> inputShapeFlt(shapeLen, 0);
    inputShapeFlt[0] = static_cast<int32_t>(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); i++) {
        inputShapeFlt[i + 1] = static_cast<int32_t>(inputShape[i]);
    }
    // calculate buffer size
    size_t dataByteCount = 0;
    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
    ONNXTensorElementDataType opDtype = AmctUtils::GetTensorEleType(api_, inputInfo);
    if (opDtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        dataByteCount = sizeof(float) * inputSize;
    } else if (opDtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        dataByteCount = sizeof(uint16_t) * inputSize;
    } else {
        LOG_ERROR("Wrong input data type. Only support float16 and float32 for dump.\n");
        return;
    }
    // dump data
    for (auto objectLayerName : objectLayerNames_) {
        this->DumpData(x, dataByteCount, inputShapeFlt, objectLayerName);
    }
}
