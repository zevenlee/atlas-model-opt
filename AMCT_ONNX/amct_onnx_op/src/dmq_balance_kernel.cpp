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
 * @file dmq_balance_kernel.cpp
 *
 * @version 1.0
 */


#include "dmq_balance_kernel.h"
#include "amct_utils.h"
#include "cast_util.h"
#include "dmq_balance.h"

DMQBalanceKernel::DMQBalanceKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "migration_strength", &migrationStrength_));

    int64_t channelNum = 0;
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "channel_num", &channelNum));
    channelNum_ = channelNum;
    objectLayerName_ = AmctUtils::GetStringAttr(api_, info, "object_layer");
    recordFileName_ = AmctUtils::GetStringAttr(api_, info, "record_file_path");
}

void DMQBalanceKernel::GetInput(OrtKernelContext* context, uint32_t index, std::vector<float> &inputData)
{
    const OrtValue* input = AmctUtils::GetKernelInput(api_, context, 0);
    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, input);
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);
    const void* data = AmctUtils::GetTensorData<void>(api_, input);

    inputData.resize(inputSize, 0);
    ONNXTensorElementDataType inputType = AmctUtils::GetTensorEleType(api_, inputInfo);
    AmctUtils::SaveInputDataToFloat32(data, inputData.data(), inputSize, inputType);
    return;
}

#if ORT_API_VERSION >= 16
OrtStatusPtr DMQBalanceKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void DMQBalanceKernel::Compute(OrtKernelContext* context)
{
    std::vector<float> dmqbFactor(channelNum_, 0);

#ifdef USE_CUDA
    const OrtValue* input0 = AmctUtils::GetKernelInput(api_, context, 0);
    OrtTensorTypeAndShapeInfo* input0Info = AmctUtils::GetTensorTypeAndShapeInfo(api_, input0);
    size_t input0Size = AmctUtils::GetElementCount(api_, input0Info);
    AmctUtils::CheckTensorNotEmpty(input0Size);
    ONNXTensorElementDataType input0Type = AmctUtils::GetTensorEleType(api_, input0Info);
    const void* data0 = AmctUtils::GetTensorData<void>(api_, input0);
    AmctCommon::InputDataParam act = {const_cast<void*>(data0), static_cast<int64_t>(input0Type), input0Size};

    const OrtValue* input1 = AmctUtils::GetKernelInput(api_, context, 1);
    OrtTensorTypeAndShapeInfo* input1Info = AmctUtils::GetTensorTypeAndShapeInfo(api_, input1);
    size_t input1Size = AmctUtils::GetElementCount(api_, input1Info);
    AmctUtils::CheckTensorNotEmpty(input1Size);
    ONNXTensorElementDataType input1Type = AmctUtils::GetTensorEleType(api_, input1Info);
    const void* data1 = AmctUtils::GetTensorData<void>(api_, input1);

    AmctCommon::InputDataParam wts = {const_cast<void*>(data1), static_cast<int64_t>(input1Type), input1Size};
    int ret = AmctCommon::DMQBalanceGpuMemCopy(act, wts, migrationStrength_, channelNum_, dmqbFactor.data());
    if (ret == AmctCommon::NOT_SUPPORT_ERROR) {
        ORT_CXX_API_THROW("AMCT cannot accept types other than float and float16.", ORT_FAIL);
    }
    if (ret != 0) {
        LOG_ERROR("Do \"%s\" DMQBalance cuda compute failed, error code: %d.\n", objectLayerName_.c_str(), ret);
        return;
    }
#else
    std::vector<float> actData;
    GetInput(context, 0, actData);
    util::FloatData act = {static_cast<unsigned int>(actData.size()), actData.data()};

    std::vector<float> wtsData;
    GetInput(context, 1, wtsData);
    util::FloatData wts = {static_cast<unsigned int>(wtsData.size()), wtsData.data()};
    int ret = AmctCommon::DMQBalance(act, wts, migrationStrength_, channelNum_, dmqbFactor.data());
    if (ret != 0) {
        LOG_ERROR("Do \"%s\" DMQBalance failed, error code: %d.\n", objectLayerName_.c_str(), ret);
        return;
    }
#endif

    ret = util::CheckBalanceFactor(dmqbFactor.data(), channelNum_);
    if (ret != AmctCommon::SUCCESS) {
        return;
    }

    std::string trimedRecordFileName = AmctUtils::TrimTailSpace(recordFileName_);
    std::string trimedLayerName = AmctUtils::TrimTailSpace(objectLayerName_);
    ret = util::RecordRepeatData(trimedRecordFileName, trimedLayerName, dmqbFactor, "tensor_balance_factor");
    if (ret != AmctCommon::SUCCESS) {
        return;
    }
}
