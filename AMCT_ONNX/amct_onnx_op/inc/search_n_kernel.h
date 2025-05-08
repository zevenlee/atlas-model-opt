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
 * @file search_n_kernel.h
 *
 * @version 1.0
 */
#ifndef SEARCH_N_KERNEL_H
#define SEARCH_N_KERNEL_H

#include "search_n.h"
#include "custom_op_library.h"

struct SearchNKernel {
public:
    SearchNKernel(const OrtApi& api, const OrtKernelInfo* info);
    ~SearchNKernel();
#if ORT_API_VERSION >= 16
    OrtStatusPtr ComputeV2(OrtKernelContext* context);
#endif
    void Compute(OrtKernelContext* context);

private:
    void RecordShiftBit(const std::vector<int>& bestN);
    Status CheckChannelNum(size_t coutNum, size_t scaleWSize, std::string layerNames);

    OrtApi api_;
    std::vector<std::vector<float>> accumulateData_{};
    int64_t batchNum_{0};
    int64_t current_batch_{0};
    std::string recordFileName_;
    std::vector<std::string> objectLayerNames_;
};

void StoreInputTensorToND(const float* inputData,
                          const size_t inputSize,
                          const std::vector<int64_t>& inputshape,
                          std::vector<std::vector<float>>& outData,
                          size_t scaleWSize);

void InitSearchnError(std::vector<std::vector<float>>& searchNError,
                      const std::vector<int64_t>& inputshape,
                      bool channelWise);

#endif // SEARCH_N_KERNEL_H