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
 * @file custom_op_library.cpp
 *
 * @version 1.0
 */
#ifndef CUSTOM_OP_LIBRARY_H
#define CUSTOM_OP_LIBRARY_H
#include <fstream>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

const unsigned int IDX_TWO = 2;
const size_t NUM_THREE = 3;
const size_t NUM_TWO = 2;
const size_t SEARCHN_INPUT_TYPE_COUNT = 3;
const size_t SEARCHNV2_INPUT_TYPE_COUNT = 3;
const size_t DMQB_INPUT_TYPE_COUNT = 2;
const size_t SUPPORT_DYN_TYPE_ORT_VERSION = 8;


struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(OrtApi ort, const OrtValue* value)
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        auto status = ort.GetTensorTypeAndShape(value, &info);
        if (status) {
            ORT_CXX_API_THROW("GetTensorTypeAndShape failed", ORT_FAIL);
        }
        if (info == nullptr) {
            ORT_CXX_API_THROW("Find nullptr in GetTensorTypeAndShape", ORT_FAIL);
        }
        size_t dimension;
        status = ort.GetDimensionsCount(info, &dimension);
        if (status) {
            ORT_CXX_API_THROW("GetDimensionsCount failed", ORT_FAIL);
        }
        std::vector<int64_t> inputShape(dimension, 0);
        status = ort.GetDimensions(info, inputShape.data(), dimension);
        if (status) {
            ORT_CXX_API_THROW("GetDimensions failed", ORT_FAIL);
        }
        std::vector<int64_t>::operator=(inputShape);
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
    ~OrtTensorDimensions() {}
};

#ifdef __cplusplus
extern "C" {
#endif

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);


#ifdef __cplusplus
}
#endif
#endif /* CUSTOM_OP_LIBRARY_H */
