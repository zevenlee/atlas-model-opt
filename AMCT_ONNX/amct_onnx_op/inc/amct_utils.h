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
 * @brief AMCT utils header file
 *
 * @file amct_utils.h
 *
 * @version 1.0
 */
#ifndef AMCT_UTILS_H
#define AMCT_UTILS_H
#include <fstream>
#include <string>
#include "custom_op_library.h"
#include "util.h"

namespace AmctUtils {
#ifdef __cplusplus
extern "C" {
#endif
void AmctDumpData(const char* filePath,
                  const int32_t* inputShapeArray,
                  int shapeLen,
                  const void* inputDataArray,
                  int dataLen);

void ConvertLayerName(std::string& originalLayerName,
                      const std::string& subString,
                      const std::string& replaceString);

std::string TrimTailSpace(std::string& str);
void CheckTensorNotEmpty(size_t tensorElementSize);
ONNXTensorElementDataType AmctOpDynamicTypeCheck();

void SaveInputDataToFloat32(const void* inputData, float* saveData, size_t dataLength, int dataId);
#ifdef __cplusplus
}
#endif

inline void CheckStatus(const OrtApi &api, OrtStatus* status)
{
    if (status != nullptr) {
        std::string errMsg(api.GetErrorMessage(status));
        ORT_CXX_API_THROW(std::move(errMsg), ORT_FAIL);
    }
}

std::string GetStringAttr(const OrtApi& api, const OrtKernelInfo* info, const std::string& attrName);

template <typename T>
inline T* GetTensorMutableData(const OrtApi& api, OrtValue* value)
{
    void* out = nullptr;
    CheckStatus(api, api.GetTensorMutableData(value, &out));
    if (out == nullptr) {
        ORT_CXX_API_THROW("OrtApi GetTensorMutableData find nullptr", ORT_FAIL);
    }
    T* result = reinterpret_cast<T*>(out);
    return result;
}

template <typename T>
const T* GetTensorData(const OrtApi& api, const OrtValue* value)
{
    return GetTensorMutableData<T>(api, const_cast<OrtValue*>(value));
}

inline std::vector<int64_t> GetShape(const OrtApi& api, OrtTensorTypeAndShapeInfo* info)
{
    size_t dimension;
    CheckStatus(api, api.GetDimensionsCount(info, &dimension));
    std::vector<int64_t> result(dimension, 0);
    CheckStatus(api, api.GetDimensions(info, result.data(), dimension));
    return result;
}

inline const OrtValue* GetKernelInput(const OrtApi& api, OrtKernelContext* context, size_t inputIdx)
{
    const OrtValue* result = nullptr;
    CheckStatus(api, api.KernelContext_GetInput(context, inputIdx, &result));
    if (result == nullptr) {
        ORT_CXX_API_THROW("Find nullptr in GetKernelInput", ORT_FAIL);
    }
    return result;
}

inline OrtValue *GetKernelOutput(
    const OrtApi &api, OrtKernelContext *context, size_t outputIdx, const int64_t *dimValues, size_t dimCount)
{
    OrtValue* result = nullptr;
    CheckStatus(api, api.KernelContext_GetOutput(context, outputIdx, dimValues, dimCount, &result));
    if (result == nullptr) {
        ORT_CXX_API_THROW("Find nullptr in GetKernelOutput", ORT_FAIL);
    }
    return result;
}

inline OrtTensorTypeAndShapeInfo* GetTensorTypeAndShapeInfo(const OrtApi &api, const OrtValue* ortValue)
{
    OrtTensorTypeAndShapeInfo* result = nullptr;
    CheckStatus(api, api.GetTensorTypeAndShape(ortValue, &result));
    if (result == nullptr) {
        ORT_CXX_API_THROW("Find nullptr in GetTensorTypeAndShapeInfo", ORT_FAIL);
    }
    return result;
}

inline ONNXTensorElementDataType GetTensorEleType(const OrtApi &api, const OrtTensorTypeAndShapeInfo* info)
{
    ONNXTensorElementDataType result;
    CheckStatus(api, api.GetTensorElementType(info, &result));
    return result;
}

inline size_t GetElementCount(const OrtApi &api, const OrtTensorTypeAndShapeInfo* info)
{
    size_t result;
    CheckStatus(api, api.GetTensorShapeElementCount(info, &result));
    return result;
}
}
#endif /* AMCT_UTILS_H */