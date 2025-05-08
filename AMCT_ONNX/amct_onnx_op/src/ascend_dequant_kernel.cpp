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
 * @file ascend_dequant_kernel.cpp
 *
 * @version 1.0
 */

#include <cmath>
#include "amct_utils.h"
#include "dequant_quant.h"
#include "ascend_dequant_kernel.h"
#include "util.h"


void GetShapeInfo(bool channelWise,
                  std::vector<int64_t> shapeInfo,
                  int64_t& hwSize,
                  int64_t& chwSize)
{
    if (channelWise) {
        for (size_t idx = 1; idx < shapeInfo.size(); ++idx) {
            chwSize *= shapeInfo[idx];
            if (idx > 1) {
                hwSize *= shapeInfo[idx];
            }
        }
    }
}

AscendDequantKernel::AscendDequantKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    auto fakeQuantPrecisionMode = AmctUtils::GetStringAttr(api_, info, "fakequant_precision_mode");
    fakeQuantPrecisionMode_ = AmctUtils::TrimTailSpace(fakeQuantPrecisionMode);
}

#if ORT_API_VERSION >= 16
OrtStatusPtr AscendDequantKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void AscendDequantKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs input 0: data
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);
    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);

    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);
    ONNXTensorElementDataType inputTensorType = AmctUtils::GetTensorEleType(api_, inputInfo);
    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);

    int64_t chwSize = 1;
    int64_t hwSize = 1;
    // input 1: shift bit
    const OrtValue* param = AmctUtils::GetKernelInput(api_, context, 1);
    const uint64_t* paramData = AmctUtils::GetTensorData<uint64_t>(api_, param);
    OrtTensorTypeAndShapeInfo* paramInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, param);
    size_t paramSize = AmctUtils::GetElementCount(api_, paramInfo);
    AmctUtils::CheckTensorNotEmpty(paramSize);

    bool channelWise = paramSize == 1 ? false : true;

    std::vector<int64_t> shapeInfo = AmctUtils::GetShape(api_, inputInfo);
    GetShapeInfo(channelWise, shapeInfo, hwSize, chwSize);
    // Setup output
    OrtTensorDimensions dimensions(api_, inputX);
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, dimensions.data(), dimensions.size());
    void* y = AmctUtils::GetTensorMutableData<void>(api_, output);

    OrtTensorTypeAndShapeInfo* outputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, output);
    ONNXTensorElementDataType outputTensorType = AmctUtils::GetTensorEleType(api_, outputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);
    //  fake dequant compute
    std::vector<float> shiftValueHost(paramSize);
    std::vector<float> deqScaleHost(paramSize);
    dequantParam_.paramSize = paramSize;
    dequantParam_.chwSize = chwSize;
    dequantParam_.hwSize = hwSize;
    dequantParam_.clipMode = CLIP_32;
    dequantParam_.paramData = paramData;
    dequantParam_.shiftValue = shiftValueHost.data();
    dequantParam_.deqScale = deqScaleHost.data();
    dequantParam_.channelWise = channelWise;
    int64_t fakePrecisionMode = 0;
    if (fakeQuantPrecisionMode_ == "FORCE_FP16_QUANT") {
        fakePrecisionMode = util::FORCE_FP16_QUANT;
    }
    InputDataParam params = {
        x, y, static_cast<int64_t>(inputTensorType), static_cast<int64_t>(outputTensorType), inputSize, fakePrecisionMode};

#ifdef USE_CUDA
    int ret = FakeDequantCuda(params, dequantParam_);
    if (ret != 0) {
        LOG_ERROR("Do AscendDequant cuda compute failed, error code: %d.\n", ret);
        return;
    }
#else
    int ret = ParseParamData(dequantParam_);
    if (ret != 0) {
        LOG_ERROR("Do ParseParamData failed, error code: %d.\n", ret);
        return;
    }
    ret = FakeDequant(params, dequantParam_);
    if (ret != 0) {
        LOG_ERROR("Do AscendDequant compute failed, error code: %d.\n", ret);
        return;
    }
#endif
}