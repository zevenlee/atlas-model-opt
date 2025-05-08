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
 * @file search_n_v2_kernel.cpp
 *
 * @version 1.0
 */

#include "amct_utils.h"
#include "search_n_v2_kernel.h"
#include "search_n_kernel.h"
#include "util.h"

using namespace util;

Status SearchNV2Kernel::CheckChannelNum(size_t coutNum, size_t scaleWSize, std::string layerNames)
{
    bool channelWise = !(scaleWSize == 1);
    if (channelWise) {
        // data has been transposed to CNHW in advance
        if (coutNum != scaleWSize) {
            LOG_ERROR("Op \"%s SearchNv2\" inputs[1]'s shape[1]{%ld} isn't equal to inputs[1]'s length {%ld}\n", \
                layerNames.c_str(), coutNum, scaleWSize);
            return AmctCommon::BAD_PARAMETERS_ERROR;
        }
    }
    return AmctCommon::SUCCESS;
}

void SearchNV2Kernel::RecordShiftBit(const std::vector<int>& bestN)
{
    for (auto objectLayerName : objectLayerNames_) {
        std::string trimedRecordFilePath = AmctUtils::TrimTailSpace(recordFileName_);
        std::string trimedObjectLayerName = AmctUtils::TrimTailSpace(objectLayerName);
        util::RecordRepeatData(trimedRecordFilePath, trimedObjectLayerName, bestN, "shift_bit");
    }
}

SearchNV2Kernel::SearchNV2Kernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "batch_num", &batchNum_));

    int64_t layerNum;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "layer_num", &layerNum));
    for (int i = 0; i < layerNum; ++i) {
        std::string attrName = "object_layer";
        attrName = attrName.append(std::to_string(i));
        std::string layerName = AmctUtils::GetStringAttr(api_, info, attrName);
        objectLayerNames_.push_back(layerName);
    }
}

SearchNV2Kernel::~SearchNV2Kernel()
{
    // Release memory used for accumulation error
    for (size_t channel = 0; channel < searchNError_.size(); channel++) {
        searchNError_.clear();
        searchNError_.shrink_to_fit();
    }
}

#if ORT_API_VERSION >= 16
OrtStatusPtr SearchNV2Kernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void SearchNV2Kernel::Compute(OrtKernelContext* context)
{
    if (++current_batch_ > batchNum_) {
        return;
    }
    // obtain input data
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);

    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
    std::vector<int64_t> inputShape = AmctUtils::GetShape(api_, inputInfo);

    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo); // CNHW
    AmctUtils::CheckTensorNotEmpty(inputSize);

    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
    std::vector<float> inData(inputSize);
    ONNXTensorElementDataType inputTypeId = AmctUtils::GetTensorEleType(api_, inputInfo);
    AmctUtils::SaveInputDataToFloat32(x, inData.data(), inputSize, inputTypeId);

    // obtain scale_d
    const OrtValue* inputScaleD = AmctUtils::GetKernelInput(api_, context, 1);

    OrtTensorTypeAndShapeInfo* scaleDInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputScaleD);
    size_t scaleDSize = AmctUtils::GetElementCount(api_, scaleDInfo);
    AmctUtils::CheckTensorNotEmpty(scaleDSize);
    if (scaleDSize != 1) {
        LOG_ERROR("SEARCHN_V2 Op \"%s\" can only have 1 scale_d, but get %zu\n",
            objectLayerNames_[0].c_str(), scaleDSize);
        return;
    }
    const float* scaleD = AmctUtils::GetTensorData<float>(api_, inputScaleD);

    // obtain scale_w
    const OrtValue* inputScaleW = AmctUtils::GetKernelInput(api_, context, 2);
    OrtTensorTypeAndShapeInfo* scaleWInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputScaleW);
    size_t scaleWSize = AmctUtils::GetElementCount(api_, scaleWInfo);
    AmctUtils::CheckTensorNotEmpty(scaleWSize);

    const float* scaleW = AmctUtils::GetTensorData<float>(api_, inputScaleW);
    // check channel
    if (CheckChannelNum(static_cast<size_t>(inputShape[0]), scaleWSize, objectLayerNames_[0]) != AmctCommon::SUCCESS) {
        return;
    }

    // calculate deqscale
    std::vector<float> deqScale;
    for (size_t i = 0; i < scaleWSize; ++i) {
        deqScale.push_back(scaleD[0] * scaleW[i]);
    }
    FloatData deqScaleCpu = {static_cast<uint>(scaleWSize), deqScale.data()};

    // store data in ND
    std::vector<std::vector<float>> currentData;
    StoreInputTensorToND(inData.data(), inputSize, inputShape, currentData, scaleWSize);

    // search_n_v2 calculate best N
    if (current_batch_ == 1) {
        bool channelWise = !(scaleWSize == 1);
        InitSearchnError(searchNError_, inputShape, channelWise);
        isBroadcast_ = (inputShape[NHWC_H_DIM] == 1) && (inputShape[NHWC_W_DIM] == 1);
    }

    if (AmctCommon::SearchNV2AccumulateError(currentData, searchNError_, deqScaleCpu, isBroadcast_) !=
        AmctCommon::SUCCESS) {
        LOG_ERROR("Layer \"%s\" SearchNV2AccumulateError failed! \n", objectLayerNames_[0].c_str());
        return;
    }

    if (current_batch_ == batchNum_) {
        std::vector<int> bestN(searchNError_.size());
        IntData bestNCpu = {static_cast<uint>(searchNError_.size()), bestN.data()};
        AmctCommon::SearchNV2FindBestNCpu(searchNError_, bestNCpu, isBroadcast_);
        // record best n
        RecordShiftBit(bestN);
    }
}
