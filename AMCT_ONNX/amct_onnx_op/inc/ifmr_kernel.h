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
 * @file ifmr_kernel.h
 *
 * @version 1.0
 */
#ifndef IFMR_KERNEL_H
#define IFMR_KERNEL_H

#include "ifmr.h"
#include "custom_op_library.h"

struct IFMRKernel {
public:
    IFMRKernel(const OrtApi& api, const OrtKernelInfo* info);
    ~IFMRKernel() {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr ComputeV2(OrtKernelContext* context);
#endif
    void Compute(OrtKernelContext* context);

private:
    void DoCalibration(float* out);
    void DumpData(const void* x, const int inputSize, const std::vector<int32_t>& inputShapeFlt,
        std::string objectLayerName);

    OrtApi api_;
    std::vector<float> accumulateData_{};
    int64_t bathNum_{0};
    int64_t currentBatch_{0};
    AmctCommon::IfmrParam ifmrParam_;
    util::FloatData scale_;
    float scaleData_{0};
    util::IntData offset_;
    int offsetData_{0};
    std::string recordFileName_;
    std::vector<std::string> objectLayerNames_;
    std::string dumpDir_;
    std::string inputStamp_ = "data";
    int opDtype_{0};
    int64_t checkCriterion;
    std::string fakeQuantPrecisionMode_;
};

#endif // IFMR_KERNEL_H