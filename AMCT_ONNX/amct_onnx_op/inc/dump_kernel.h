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
 * @file dump_kernel.h
 *
 * @version 1.0
 */
#ifndef DUMP_KERNEL_H
#define DUMP_KERNEL_H

#include "custom_op_library.h"

struct DUMPKernel {
public:
    DUMPKernel(const OrtApi& api, const OrtKernelInfo* info);
    ~DUMPKernel() {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr ComputeV2(OrtKernelContext* context);
#endif
    void Compute(OrtKernelContext* context);

private:
    void DumpData(const void* x, const int inputSize, const std::vector<int32_t>& inputShapeFlt,
        std::string objectLayerName);
    OrtApi api_;
    int64_t bathNum_{0};
    int64_t currentBatch_{0};
    std::string recordFileName_;
    std::vector<std::string> objectLayerNames_;
    std::string dumpDir_;
    std::string dumpStamp_;
};

#endif // DUMP_KERNEL_H