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
 * @file search_n_kernel.cpp
 *
 * @version 1.0
 */

#include "search_n_kernel.h"
#include <string>
#include <cmath>

#include "amct_utils.h"
#include "util.h"

using namespace util;

void InitSearchnError(std::vector<std::vector<float>>& searchNError,
                      const std::vector<int64_t>& inputshape,
                      bool channelWise)
{
    size_t channelNum = channelWise ? inputshape[0] : 1;
    for (size_t channel = 0; channel < channelNum; channel++) {
        std::vector<float> error(SHIFT_BITS);
        searchNError.push_back(error);
    }
}

void StoreInputTensorToND(const float* inputData,
                          const size_t inputSize,
                          const std::vector<int64_t>& inputshape,
                          std::vector<std::vector<float>>& outData,
                          size_t scaleWSize)
{
    bool channelWise = !(scaleWSize == 1);

    if (!channelWise) {
        // not channel wise
        if (outData.size() == 0) {
            std::vector<float> data;
            data.insert(data.end(), inputData, inputData + inputSize);
            outData.push_back(data);
        } else {
            outData[0].insert(outData[0].end(), inputData, inputData + inputSize);
        }
    } else {
        // channel wise
        int channelSize = 1;
        for (size_t i = 1; i < inputshape.size(); i++) {
            channelSize = channelSize * inputshape[i];
        }
        // store data by channel
        for (uint channel = 0; channel < inputshape[0]; channel++) {
            if (outData.size() == channel) {
                std::vector<float> data;
                data.insert(data.end(), inputData, inputData + channelSize);
                outData.push_back(data);
            } else {
                outData[channel].insert(outData[channel].end(), inputData, inputData + channelSize);
            }
            inputData = inputData + channelSize;
        }
    }
}

Status SearchNKernel::CheckChannelNum(size_t coutNum, size_t scaleWSize, std::string layerNames)
{
    bool channelWise = (scaleWSize != 1);
    if (channelWise) {
        // data has been transposed to CNHW in advance
        if (coutNum != scaleWSize) {
            LOG_ERROR("Op \"%s SearchN\" inputs[1]'s shape[1]{%ld} isn't equal to inputs[1]'s length {%ld}\n", \
                layerNames.c_str(), coutNum, scaleWSize);
            return AmctCommon::BAD_PARAMETERS_ERROR;
        }
    }
    return AmctCommon::SUCCESS;
}

SearchNKernel::SearchNKernel(const OrtApi& api, const OrtKernelInfo* info)
    : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "batch_num", &batchNum_));
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");

    int64_t layerNum;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "layer_num", &layerNum));

    for (int i = 0; i < layerNum; ++i) {
        std::string attrName = "object_layer";
        attrName = attrName.append(std::to_string(i));
        std::string layerName = AmctUtils::GetStringAttr(api_, info, attrName);
        objectLayerNames_.push_back(layerName);
    }
}

void SearchNKernel::RecordShiftBit(const std::vector<int>& bestN)
{
    for (auto objectLayerName : objectLayerNames_) {
        std::string trimedRecordFilePath = AmctUtils::TrimTailSpace(recordFileName_);
        std::string trimedObjectLayerName = AmctUtils::TrimTailSpace(objectLayerName);
        util::RecordRepeatData(trimedRecordFilePath, trimedObjectLayerName, bestN, "shift_bit");
    }
}

SearchNKernel::~SearchNKernel()
{
    // Release memory used for data accumulation
    for (size_t channel = 0; channel < accumulateData_.size(); channel++) {
        accumulateData_.clear();
        accumulateData_.shrink_to_fit();
    }
}

#if ORT_API_VERSION >= 16
OrtStatusPtr SearchNKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void SearchNKernel::Compute(OrtKernelContext* context)
{
    if (++current_batch_ > batchNum_) {
        return;
    }

    // get scale_w
    const OrtValue* inputScaleW = AmctUtils::GetKernelInput(api_, context, 2);
    if (inputScaleW == nullptr) {
        ORT_CXX_API_THROW("Find nullptr inputScaleW", ORT_FAIL);
    }
    OrtTensorTypeAndShapeInfo* scaleWInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputScaleW);
    if (scaleWInfo == nullptr) {
        ORT_CXX_API_THROW("Find nullptr scaleWInfo", ORT_FAIL);
    }
    size_t scaleWSize = AmctUtils::GetElementCount(api_, scaleWInfo);

    const float* scaleW = AmctUtils::GetTensorData<float>(api_, inputScaleW);
    // accumulate data
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);
    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
    std::vector<int64_t> inputShape = AmctUtils::GetShape(api_, inputInfo);
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);

    if (CheckChannelNum(static_cast<size_t>(inputShape[0]), scaleWSize, objectLayerNames_[0]) != AmctCommon::SUCCESS) {
        return;
    }
    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);

    std::vector<float> inData(inputSize);
    ONNXTensorElementDataType inputTypeId = AmctUtils::GetTensorEleType(api_, inputInfo);
    AmctUtils::SaveInputDataToFloat32(x, inData.data(), inputSize, inputTypeId);

    // store data in ND
    StoreInputTensorToND(inData.data(), inputSize, inputShape, accumulateData_, scaleWSize);

    if (current_batch_ != batchNum_) {
        return;
    }
    // get scale_d
    const OrtValue* inputScaleD = AmctUtils::GetKernelInput(api_, context, 1);
    OrtTensorTypeAndShapeInfo* scaleDInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputScaleD);
    size_t scaleDSize = AmctUtils::GetElementCount(api_, scaleDInfo);
    if (scaleDSize != 1) {
        LOG_ERROR("SearchN Op \"%s\" can only have 1 scale_d, but get %zu\n",
            objectLayerNames_[0].c_str(),
            scaleDSize);
        ORT_CXX_API_THROW("AMCT searchN op failed", ORT_FAIL);
    }

    const float* scaleD = AmctUtils::GetTensorData<float>(api_, inputScaleD);
    std::vector<float> deqScale;
    for (size_t i = 0; i < scaleWSize; ++i) {
        deqScale.push_back(scaleD[0] * scaleW[i]);
    }
    std::vector<std::vector<int>> int32Data(accumulateData_.size(), std::vector<int>(accumulateData_[0].size(), 0));
    for (size_t i = 0; i < accumulateData_.size(); ++i) {
        // prevent divide by zero. a number less than epsilon means that it can be treated as zero, but still
        // divisible by it.
        if (deqScale[i] == 0.0) {
            ORT_CXX_API_THROW("AMCT searchN op get zero deqScale", ORT_FAIL);
        }
        for (size_t j = 0; j < accumulateData_[i].size(); ++j) {
            int32Data[i][j] = round(accumulateData_[i][j] / deqScale[i]);
        }
    }

    std::vector<int> bestN;
    AmctCommon::SearchShiftBits(int32Data, bestN);
    RecordShiftBit(bestN);
}
