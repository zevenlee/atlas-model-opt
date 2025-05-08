/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
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
 * @file ifmr_kernel.cpp
 *
 * @version 1.0
 */

#include <sstream>

#include "amct_utils.h"
#include "ifmr_kernel.h"
#include "util.h"
#include "cast_util.h"

IFMRKernel::IFMRKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    inputStamp_ = AmctUtils::GetStringAttr(api_, info, "input_stamp");
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "batch_num", &bathNum_));
    int64_t numBits = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "num_bits", &numBits));
    ifmrParam_.numBits = numBits;
    int64_t withOffset = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "with_offset", &withOffset));
    ifmrParam_.withOffset = withOffset;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "start_ratio", &ifmrParam_.startRatio));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "end_ratio", &ifmrParam_.endRatio));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "step", &ifmrParam_.step));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "max_percentile", &ifmrParam_.maxPercentile));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "min_percentile", &ifmrParam_.minPercentile));
    int64_t needDump = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "need_dump", &needDump));
    ifmrParam_.needDump = needDump;
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "check_criterion", &checkCriterion));
    dumpDir_ = AmctUtils::GetStringAttr(api_, info, "dump_dir");

    auto fakeQuantPrecisionMode = AmctUtils::GetStringAttr(api_, info, "fakequant_precision_mode");
    fakeQuantPrecisionMode_ = AmctUtils::TrimTailSpace(fakeQuantPrecisionMode);

    int64_t layerNum;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "layer_num", &layerNum));
    for (int i = 0; i < layerNum; ++i) {
        std::string attrName = "object_layer";
        attrName = attrName.append(std::to_string(i));
        std::string layerName = AmctUtils::GetStringAttr(api_, info, attrName);
        objectLayerNames_.push_back(AmctUtils::TrimTailSpace(layerName));
    }

    scale_.length = 1;
    scale_.data = &scaleData_;
    offset_.length = 1;
    offset_.data = &offsetData_;
}

void IFMRKernel::DumpData(const void* x, const int inputSize, const std::vector<int32_t>& inputShapeFlt,
    std::string objectLayerName)
{
    std::string trimedLayerName = AmctUtils::TrimTailSpace(objectLayerName);
    std::string trimedDumpDir = AmctUtils::TrimTailSpace(dumpDir_);
    AmctUtils::ConvertLayerName(trimedLayerName, "/", "_");
    std::stringstream ss;
    ss << trimedDumpDir << '/' << trimedLayerName << \
        "_act_calibration_layer_" << std::to_string(currentBatch_) << ".bin";
    std::string fileName = ss.str();
    AmctUtils::AmctDumpData(fileName.c_str(), inputShapeFlt.data(), inputShapeFlt.size(), x, inputSize);
}

#if ORT_API_VERSION >= 16
OrtStatusPtr IFMRKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void IFMRKernel::Compute(OrtKernelContext* context)
{
    // set output
    std::vector<int64_t> outputShape = {1};
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, outputShape.data(), outputShape.size());
    float* outFp32 = AmctUtils::GetTensorMutableData<float>(api_, output);
    outFp32[0] = scaleData_;

    // Setup inputs
    currentBatch_++;
    if (currentBatch_ > bathNum_) {
        return;
    }
    // accumulate data
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);

    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);

    std::vector<int64_t> inputShape = AmctUtils::GetShape(api_, inputInfo);
    if (checkCriterion == 1 && inputShape[0] != 1) {
        std::string errMsg = \
            "Node " + objectLayerNames_[0] + " cannot be quantize for its sequence_lens is bigger than 1";
        ORT_CXX_API_THROW(errMsg.c_str(), ORT_FAIL);
    }
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    size_t dataByteCount = sizeof(float) * inputSize;
    AmctUtils::CheckTensorNotEmpty(inputSize);

    size_t shapeLen = inputShape.size() + 1;
    std::vector<int32_t> inputShapeFlt(shapeLen, 0);
    inputShapeFlt[0] = static_cast<int32_t>(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); i++) {
        inputShapeFlt[i + 1] = static_cast<int32_t>(inputShape[i]);
    }

    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
    ONNXTensorElementDataType opDtype = AmctUtils::GetTensorEleType(api_, inputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(inputInfo);
    opDtype_ = static_cast<int64_t>(opDtype);
    if (opDtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        dataByteCount = sizeof(uint16_t) * inputSize;
    }
    size_t dataOffset = accumulateData_.size();
    accumulateData_.resize(dataOffset + inputSize);
    AmctUtils::SaveInputDataToFloat32(x, accumulateData_.data() + dataOffset, inputSize, opDtype_);

    for (auto objectLayerName : objectLayerNames_) {
        if (ifmrParam_.needDump) {
            this->DumpData(x, dataByteCount, inputShapeFlt, objectLayerName);
        }
    }
    if (currentBatch_ != bathNum_) {
        return;
    }
    DoCalibration(outFp32);
}

void IFMRKernel::DoCalibration(float* out)
{
    // start to do ifmr calibration
    ifmrParam_.calibration = 0;
    ifmrParam_.needDump = false;
    int ret = AmctCommon::IfmrQuant(accumulateData_.data(), accumulateData_.size(), ifmrParam_, scale_, offset_);
    if (ret != 0) {
        LOG_ERROR("Do IFMR calibration failed, error code: %d.\n", ret);
        ORT_CXX_API_THROW("Do IFMR calibration failed", ORT_FAIL);
    }
    out[0] = scaleData_;
    std::string trimedRecordFilePath = AmctUtils::TrimTailSpace(recordFileName_);
    for (auto objectLayerName : objectLayerNames_) {
        std::string trimedObjectLayerName = AmctUtils::TrimTailSpace(objectLayerName);
        std::string trimedInputSign_ = AmctUtils::TrimTailSpace(inputStamp_);
        if (trimedInputSign_ == "weight" || trimedInputSign_ == "initial_h") {
            opDtype_ = 0;
        }
        util::RecordData<int> recordData = {
            scaleData_, offsetData_, {}, trimedInputSign_, opDtype_, ifmrParam_.numBits, fakeQuantPrecisionMode_};
        util::RecordScaleOffset(trimedRecordFilePath, trimedObjectLayerName, recordData);
    }
}
