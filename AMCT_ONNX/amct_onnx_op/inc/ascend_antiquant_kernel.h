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
 * @file ascend_antiquant_kernel.h
 *
 * @version 1.0
 */
#ifndef ASCEND_ANTIQUANT_KERNEL_H
#define ASCEND_ANTIQUANT_KERNEL_H

#include "custom_op_library.h"

struct AntiQuantKernel {
public:
    AntiQuantKernel(const OrtApi& api, const OrtKernelInfo* info);
    ~AntiQuantKernel() {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr ComputeV2(OrtKernelContext* context);
#endif
    void Compute(OrtKernelContext* context);

private:
    OrtApi api_;
    float scaleData_{0};
    float offsetData_{0};
    int64_t quantBits_{0};
};

#endif // ASCEND_ANTIQUANT_KERNEL_H