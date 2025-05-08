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
 * @file dmq_balance_kernel.h
 *
 * @version 1.0
 */
#ifndef DMQ_BALANCE_KERNEL_H
#define DMQ_BALANCE_KERNEL_H

#include "custom_op_library.h"

struct DMQBalanceKernel {
public:
    DMQBalanceKernel(const OrtApi& api, const OrtKernelInfo* info);
    ~DMQBalanceKernel() {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr ComputeV2(OrtKernelContext* context);
#endif
    void Compute(OrtKernelContext* context);

private:
    void GetInput(OrtKernelContext* context, uint32_t index, std::vector<float> &inputData);
    OrtApi api_;
    float migrationStrength_{0};
    uint32_t channelNum_{0};
    std::string objectLayerName_;
    std::string recordFileName_;
};

#endif // DMQ_BALANCE_KERNEL_H
