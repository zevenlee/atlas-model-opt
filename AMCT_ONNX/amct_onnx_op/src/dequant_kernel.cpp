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
 * @file dequant_kernel.cpp
 *
 * @version 1.0
 */

#include <cmath>
#include "amct_utils.h"
#include "dequant_quant.h"
#include "dequant_kernel.h"
#include "util.h"

DequantKernel::DequantKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "clip_mode", &dequantParam_.clipMode));
}

#if ORT_API_VERSION >= 16
OrtStatusPtr DequantKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void DequantKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs input 0: data
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);

    const void* inputData = AmctUtils::GetTensorData<void>(api_, inputX);
    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
    ONNXTensorElementDataType inputTensorType = AmctUtils::GetTensorEleType(api_, inputInfo);
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);

    std::vector<int64_t> shapeInfo = AmctUtils::GetShape(api_, inputInfo);
    int64_t chwSize = 1;
    int64_t hwSize = 1;
    // input 1: shift bit
    const OrtValue* shiftValue = AmctUtils::GetKernelInput(api_, context, 1);
    const float* shiftData = AmctUtils::GetTensorData<float>(api_, shiftValue);
    // input 2: deqScale
    const OrtValue* deqScale = AmctUtils::GetKernelInput(api_, context, 2);
    const float* deqScaleData = AmctUtils::GetTensorData<float>(api_, deqScale);
    OrtTensorTypeAndShapeInfo* deqScaleInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, deqScale);

    size_t deqScaleSize = AmctUtils::GetElementCount(api_, deqScaleInfo);
    AmctUtils::CheckTensorNotEmpty(deqScaleSize);
    bool channelWise = deqScaleSize == 1 ? false : true;
    if (channelWise) {
        for (size_t idx = 1; idx < shapeInfo.size(); ++idx) {
            chwSize *= shapeInfo[idx];
            if (idx > 1) {
                hwSize *= shapeInfo[idx];
            }
        }
    }

    std::vector<float> shiftDataWithPow(deqScaleSize);
    std::vector<float> deqScaleDataTmp(deqScaleSize);
    for (size_t idx = 0; idx < deqScaleSize; idx++) {
        shiftDataWithPow[idx] = pow(NUM_TWO, shiftData[idx]);
        deqScaleDataTmp[idx] = deqScaleData[idx];
    }
    // Setup output
    OrtTensorDimensions dimensions(api_, inputX);
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, dimensions.data(), dimensions.size());
    auto outputData = AmctUtils::GetTensorMutableData<void>(api_, output);
    OrtTensorTypeAndShapeInfo* outputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, output);
    ONNXTensorElementDataType outputTensorType = AmctUtils::GetTensorEleType(api_, outputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);
    //  fake dequant compute
    dequantParam_.chwSize = chwSize;
    dequantParam_.hwSize = hwSize;
    dequantParam_.shiftValue = shiftDataWithPow.data();
    dequantParam_.deqScale = deqScaleDataTmp.data();
    dequantParam_.channelWise = channelWise;
    InputDataParam params = {inputData,
        outputData,
        static_cast<int64_t>(inputTensorType),
        static_cast<int64_t>(outputTensorType),
        inputSize};
    int ret = FakeDequant(params, dequantParam_);
    if (ret != 0) {
        LOG_ERROR("Do dequant compute failed, error code: %d.\n", ret);
        return;
    }
}
