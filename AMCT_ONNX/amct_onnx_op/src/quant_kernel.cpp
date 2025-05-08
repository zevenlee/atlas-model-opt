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
 * @file quant_kernel.cpp
 *
 * @version 1.0
 */

#include "amct_utils.h"
#include "dequant_quant.h"
#include "quant_kernel.h"
#include "util.h"

QuantKernel::QuantKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "scale", &scaleData_));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "offset", &offsetData_));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_int64(info, "quant_bit", &quantBits_));
}

#if ORT_API_VERSION >= 16
OrtStatusPtr QuantKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void QuantKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs
    const OrtValue* input = AmctUtils::GetKernelInput(api_, context, 0);

    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, input);

    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    ONNXTensorElementDataType inputTensorType = AmctUtils::GetTensorEleType(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);

    const void* inputData = AmctUtils::GetTensorData<void>(api_, input);
    // Setup output
    OrtTensorDimensions dimensions(api_, input);
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, dimensions.data(), dimensions.size());
    void* outputData = AmctUtils::GetTensorMutableData<void>(api_, output);
    OrtTensorTypeAndShapeInfo* outputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, output);
    ONNXTensorElementDataType outputTensorType = AmctUtils::GetTensorEleType(api_, outputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);
    InputDataParam params = {inputData,
        outputData,
        static_cast<int64_t>(inputTensorType),
        static_cast<int64_t>(outputTensorType),
        inputSize};

#ifdef USE_CUDA
    // Launch on stream 0 or user provided stream
    int ret = FakeQuantCuda(params, quantBits_, scaleData_, offsetData_);
    if (ret != 0) {
        LOG_ERROR("Do quant cuda compute failed, error code: %d.\n", ret);
        return;
    }
#else
    int ret = FakeQuant(params, quantBits_, scaleData_, offsetData_);
    if (ret != 0) {
        LOG_ERROR("Do quant compute failed, error code: %d.\n", ret);
        return;
    }
#endif
}
