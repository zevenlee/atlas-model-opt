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
 * @file hfmg_kernel.cpp
 *
 * @version 1.0
 */

#include <sstream>
#include "amct_utils.h"
#include "hfmg_kernel.h"
#include "util.h"
#include "cast_util.h"

using namespace util;

HFMGKernel::HFMGKernel(const OrtApi& api, const OrtKernelInfo* info)
    : api_(api)
{
    inputStamp_ = AmctUtils::GetStringAttr(api_, info, "input_stamp");
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "batch_num", &bathNum_));
    int64_t numBits = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "num_bits", &numBits));
    hfmgAlgoParam_.quantBitNum = numBits;
    int64_t withOffset = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "with_offset", &withOffset));
    hfmgAlgoParam_.withOffset = withOffset;
    int64_t nbins = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "nbins", &nbins));
    hfmgAlgoParam_.nbins = nbins;
    int64_t needDump = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "need_dump", &needDump));
    needDump_ = needDump;
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "check_criterion", &checkCriterion));
    int64_t layerNum;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "layer_num", &layerNum));
    for (int i = 0; i < layerNum; ++i) {
        std::string attrName = "object_layer";
        attrName = attrName.append(std::to_string(i));
        std::string layerName = AmctUtils::GetStringAttr(api_, info, attrName);
        objectLayerNames_.push_back(AmctUtils::TrimTailSpace(layerName));
    }
    dumpDir_ = AmctUtils::GetStringAttr(api_, info, "dump_dir");

    auto fakeQuantPrecisionMode = AmctUtils::GetStringAttr(api_, info, "fakequant_precision_mode");
    fakeQuantPrecisionMode_ = AmctUtils::TrimTailSpace(fakeQuantPrecisionMode);

    scale_.data = &scaleData_;
    offset_.length = 1;
    offset_.data = &offsetData_;
}

void HFMGKernel::UpdateMinMax(const float* inputData, const int count, float& min, float& max)
{
    float inputMin = *std::min_element(inputData, inputData + count);
    float inputMax = *std::max_element(inputData, inputData + count);
    min = inputMin < min ? inputMin : min;
    max = inputMax > max ? inputMax : max;
}


void HFMGKernel::DumpData(const void* x, const int inputSize, const std::vector<int32_t>& inputShapeFlt)
{
    for (auto objectLayerName : objectLayerNames_) {
        if (needDump_) {
            std::string trimedLayerName_ = AmctUtils::TrimTailSpace(objectLayerName);
            std::string trimedDumpDir_ = AmctUtils::TrimTailSpace(dumpDir_);
            AmctUtils::ConvertLayerName(trimedLayerName_, "/", "_");
            std::stringstream ss;
            ss << trimedDumpDir_ << '/' << trimedLayerName_ << \
                "_act_calibration_layer_" << std::to_string(currentBatch_) << ".bin";
            std::string fileName = ss.str();
            AmctUtils::AmctDumpData(fileName.c_str(), inputShapeFlt.data(), inputShapeFlt.size(), x, inputSize);
        }
    }
}


int HFMGKernel::Accumlate(OrtKernelContext* context)
{
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
    // dump the input data each batch
    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
    const float* floatX = reinterpret_cast<const float*>(x);
    this->DumpData(x, dataByteCount, inputShapeFlt);
    ONNXTensorElementDataType inputType = AmctUtils::GetTensorEleType(api_, inputInfo);
    inputTypeId_ = static_cast<int64_t>(inputType);
    if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        this->DumpData(x, dataByteCount, inputShapeFlt);
        AmctCommon::InputData<float> inputData{static_cast<unsigned int>(inputSize), floatX};
        int ret = HfmgMerge(hfmgAlgoParam_.nbins, dataBins_, inputData);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("Do HfmgMerge error, error code is %d", ret);
        }
        return ret;
    }
    std::vector<float> accumulateData(inputSize);
    AmctUtils::SaveInputDataToFloat32(x, accumulateData.data(), inputSize, inputType);

    dataByteCount = sizeof(uint16_t) * inputSize;
    this->DumpData(x, dataByteCount, inputShapeFlt);
    AmctCommon::InputData<float> inputData{static_cast<unsigned int>(inputSize), accumulateData.data()};
    int ret = HfmgMerge(hfmgAlgoParam_.nbins, dataBins_, inputData);
    if (ret != AmctCommon::SUCCESS) {
        LOG_ERROR("Do HfmgMerge error, error code is %d", ret);
    }

    return ret;
}

#if ORT_API_VERSION >= 16
OrtStatusPtr HFMGKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void HFMGKernel::Compute(OrtKernelContext* context)
{
    // set output
    std::vector<int64_t> outputDims = {1};
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, outputDims.data(), outputDims.size());
    float* outFp32 = AmctUtils::GetTensorMutableData<float>(api_, output);
    outFp32[0] = scaleData_;
    // Setup inputs
    currentBatch_++;
    if (currentBatch_ > bathNum_) {
        return;
    }
    // accumulate data to Histogram and dump the input data each batch
    if (this->Accumlate(context) != AmctCommon::SUCCESS) {
        LOG_ERROR("HFMGKernel::Accumlate fail");
        return;
    }

    if (currentBatch_ == bathNum_) {
        // start to do hfmg calibration
        int ret = AmctCommon::HfmgCompute(dataBins_, scaleData_, offsetData_, hfmgAlgoParam_);
        if (ret != AmctCommon::SUCCESS) {
            LOG_ERROR("Do HfmgCompute calculate scale and offset error, error code is %d", ret);
            return;
        }
        std::string trimedRecordFilePath = AmctUtils::TrimTailSpace(recordFileName_);
        for (auto objectLayerName : objectLayerNames_) {
            std::string trimedObjectLayerName = AmctUtils::TrimTailSpace(objectLayerName);
            std::string trimedInputSign_ = AmctUtils::TrimTailSpace(inputStamp_);
            if (trimedInputSign_ == "weight" || trimedInputSign_ == "initial_h") {
                inputTypeId_ = 0;
            }
            util::RecordData<int> recordData = {
                scaleData_, offsetData_, {}, trimedInputSign_, inputTypeId_, hfmgAlgoParam_.quantBitNum,
                fakeQuantPrecisionMode_};
            util::RecordScaleOffset(trimedRecordFilePath, trimedObjectLayerName, recordData);
        }
    } else {
        const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);
        OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
        size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);

        const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
        std::vector<float> currentData(inputSize);
        AmctUtils::SaveInputDataToFloat32(x, currentData.data(), inputSize, inputTypeId_);

        float currentMin = 0;
        float currentMax = 0;
        UpdateMinMax(currentData.data(), inputSize, currentMin, currentMax);
        FloatData scaleData = {1, &scaleData_};
        IntData offsetData = {1, &offsetData_};
        AmctCommon::ActArqCalibration(currentMin, currentMax, scaleData, offsetData, hfmgAlgoParam_);
    }
    outFp32[0] = scaleData_;
}
