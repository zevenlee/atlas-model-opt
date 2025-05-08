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
 * @file ascend_quant_kernel.cpp
 *
 * @version 1.0
 */

#include "amct_utils.h"
#include "cast_util.h"
#include "dequant_quant.h"
#include "ascend_quant_kernel.h"
#include "util.h"
#include <vector>

using namespace util;

AscendQuantKernel::AscendQuantKernel(const OrtApi &api, const OrtKernelInfo *info) : api_(api)
{
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "scale", &scaleData_));
    AmctUtils::CheckStatus(api_, api_.KernelInfoGetAttribute_float(info, "offset", &offsetData_));
    dstType_ = AmctUtils::GetStringAttr(api_, info, "dst_type");
    auto fakeQuantPrecisionMode = AmctUtils::GetStringAttr(api_, info, "fakequant_precision_mode");
    fakeQuantPrecisionMode_ = AmctUtils::TrimTailSpace(fakeQuantPrecisionMode);
}

#if ORT_API_VERSION >= 16
OrtStatusPtr AscendQuantKernel::ComputeV2(OrtKernelContext* context)
{
    Compute(context);
    return api_.CreateStatus(ORT_OK, "Success");
}
#endif

void AscendQuantKernel::Compute(OrtKernelContext* context)
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
    OrtTensorTypeAndShapeInfo* outputInfo = AmctUtils::GetTensorTypeAndShapeInfo(api_, output);
    ONNXTensorElementDataType outputTensorType = AmctUtils::GetTensorEleType(api_, outputInfo);
    api_.ReleaseTensorTypeAndShapeInfo(outputInfo);
    void* y = AmctUtils::GetTensorMutableData<void>(api_, output);

    int64_t fakePrecisionMode = 0;
    if (fakeQuantPrecisionMode_ == "FORCE_FP16_QUANT") {
        fakePrecisionMode = util::FORCE_FP16_QUANT;
    }
    InputDataParam params = {
        x, y, static_cast<int64_t>(inputTensorType), static_cast<int64_t>(outputTensorType), inputSize, fakePrecisionMode};

    int64_t offsetData = static_cast<int64_t>(offsetData_);
    auto trimedDstType = AmctUtils::TrimTailSpace(dstType_);
    auto quantBitsItem = dstType2QuantBits_.find(trimedDstType);
    if (quantBitsItem == dstType2QuantBits_.end()) {
        LOG_ERROR("Cannot support AscendQuant with \"dst_type\": %s.\n", trimedDstType.c_str());
        return;
    }
    int64_t quantBits = quantBitsItem->second;

#ifdef USE_CUDA
    // Launch on stream 0 or user provided stream
    int ret = FakeQuantCuda(params, quantBits, scaleData_, offsetData);
    if (ret != 0) {
        LOG_ERROR("Do AscendQuant cuda compute failed, error code: %d.\n", ret);
        return;
    }
#else
    int ret = FakeQuant(params, quantBits, scaleData_, offsetData);
    if (ret != 0) {
        LOG_ERROR("Do AscendQuant compute failed, error code: %d.\n", ret);
        return;
    }
#endif
}