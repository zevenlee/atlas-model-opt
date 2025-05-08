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
 * @file ascend_antiquant_kernel.cpp
 *
 * @version 1.0
 */


#include "amct_utils.h"
#include "dequant_quant.h"
#include "ascend_antiquant_kernel.h"
#include "util.h"
#include "cast_util.h"

AntiQuantKernel::AntiQuantKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "scale", &scaleData_));
}

#if ORT_API_VERSION >= 16
OrtStatusPtr AntiQuantKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void AntiQuantKernel::Compute(OrtKernelContext* context)
{
    // Setup inputs
    const OrtValue* inputX = AmctUtils::GetKernelInput(api_, context, 0);
    OrtTensorTypeAndShapeInfo* inputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, inputX);

    size_t inputSize = AmctUtils::GetElementCount(api_, inputInfo);
    AmctUtils::CheckTensorNotEmpty(inputSize);

    const void* x = AmctUtils::GetTensorData<void>(api_, inputX);
    ONNXTensorElementDataType inputTensorType = AmctUtils::GetTensorEleType(api_, inputInfo);

    // Setup output
    OrtTensorDimensions dimensions(api_, inputX);
    OrtValue* output = AmctUtils::GetKernelOutput(api_, context, 0, dimensions.data(), dimensions.size());
    void* y = AmctUtils::GetTensorMutableData<void>(api_, output);

    OrtTensorTypeAndShapeInfo* outputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, output);
    ONNXTensorElementDataType outputTensorType = AmctUtils::GetTensorEleType(api_, outputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);

    InputDataParam params = {
        x, y, static_cast<int64_t>(inputTensorType), static_cast<int64_t>(outputTensorType), inputSize};
    // do real calculate
#ifdef USE_CUDA
    auto ret = FakeAntiQuantCuda(params, scaleData_);
    if (ret != 0) {
        LOG_ERROR("Do AscendAntiquant cuda compute failed, error code: %d.\n", ret);
        return;
    }
#else
    auto ret = FakeAntiQuant(params, scaleData_);
    if (ret != 0) {
        LOG_ERROR("Do AscendAntiquant compute failed, error code: %d.\n", ret);
        return;
    }
#endif
}