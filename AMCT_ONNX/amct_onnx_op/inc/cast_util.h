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
 * @brief cast_util head file
 *
 * @file cast_util.h in common_cpp
 *
 * @version 1.0
 */

#ifndef CAST_UTIL_H
#define CAST_UTIL_H

#include <cstdint>

namespace util {

constexpr uint32_t FP16_SIGN_SHIFT = 15;
constexpr uint32_t FP16_EXP_SHIFT = 10;
constexpr uint32_t FP32_SIGN_SHIFT = 31;
constexpr uint32_t FP32_EXP_SHIFT = 23;
constexpr uint32_t FP32_FRAC_SHIFT = 13;
constexpr uint32_t FP32_DENORMAL_EXP = 127 - 14;
constexpr uint32_t FP32_NORMAL_EXP = 127 - 15;

union CastTransData {
    float x;
    uint32_t y;
};

float Fp16ToFp32(uint16_t inputData);
uint16_t Fp32ToFp16(float inputData);
float CastToFP16PrecisionCPU(float inputData);
float CastToS19CPU(float data);
float FakeFp16PrecisionDataCPU(float data);

template <typename Device, typename T>
struct DataCastToFloat32Functor {
    void operator()(const T* in, float* out, int length) const;
};

template <typename Device, typename T>
struct DataCastToFloat16Functor {
    void operator()(const T* in, uint16_t* out, int length) const;
};

template <typename Device, typename T>
struct DataCastToFp16Precision {
    void operator()(const T* in, float* out, int length) const;
};

}

#endif // CAST_UTIL_H
