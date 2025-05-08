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
 * @brief fake_dequant_quant custom op kernel func C++ implement
 *
 * @file dequant_quant.cpp
 *
 * @version 1.0
 */
#include <cmath>
#include <mutex>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

#include "dequant_quant.h"
#include "cast_util.h"
#include "util.h"

using namespace std;
using namespace util;

constexpr int SHIFT_VAL_LEN = 32;
constexpr int FLOAT_TYPE_ID = 1;
constexpr int INT8_TYPE_ID = 3;
constexpr int FLOAT16_TYPE_ID = 10;
constexpr int QUANT_BIT_NUM = 8;


template<class T>
Status FakeDequantKernel(const T* data,
                         T* outputData,
                         int64_t length,
                         DequantParam dequantParam,
                         int64_t fakePrecisionMode)
{
    T clipMin = -static_cast<T>(pow(BINARY_BASE, dequantParam.clipMode - 1));
    T clipMax = static_cast<T>(pow(BINARY_BASE, dequantParam.clipMode - 1) - 1);
    if (dequantParam.chwSize == 0 || dequantParam.hwSize == 0) {
        return AmctCommon::GENERIC_ERROR;
    }
#pragma omp parallel for
    for (int64_t index = 0; index < length; index++) {
        int channelIndex = !dequantParam.channelWise ? 0 : (index % (dequantParam.chwSize)) / dequantParam.hwSize;
        float shiftValuePow = dequantParam.shiftValue[channelIndex];
        T tmpData = 0;
        if (std::fabs(shiftValuePow - 1) <= std::numeric_limits<float>::epsilon()) {
            tmpData = data[index];
        } else {
            tmpData = floor(data[index] / shiftValuePow);
        }
        if (tmpData < clipMin) {
            tmpData = clipMin;
        } else if (tmpData > clipMax) {
            tmpData = clipMax;
        }
        if (fakePrecisionMode == util::FORCE_FP16_QUANT) {
            outputData[index] = util::CastToFP16PrecisionCPU(tmpData * util::CastToS19CPU(dequantParam.deqScale[channelIndex]) * shiftValuePow);
        } else {
            outputData[index] = tmpData * dequantParam.deqScale[channelIndex] * shiftValuePow;
        }
    }
    return AmctCommon::SUCCESS;
}


template<class T>
Status FakeQuantKernel(const T* inputData,
                       T* outputData,
                       int64_t length,
                       int64_t quantBits,
                       FakeCalParams calParams)
{
    int64_t clipMin = -pow(BINARY_BASE, quantBits - 1);
    int64_t clipMax = pow(BINARY_BASE, quantBits - 1) - 1;

    auto quantFunc = [calParams](T x) {
        if (calParams.fakePrecisonMode == util::FORCE_FP16_QUANT) {
            return rint(util::CastToFP16PrecisionCPU(util::CastToFP16PrecisionCPU(
                (util::CastToFP16PrecisionCPU(x) * util::FakeFp16PrecisionDataCPU(calParams.scale))) + calParams.offset));
        } else {
            return rint(x * calParams.scale) + calParams.offset;
        }
    };

#pragma omp parallel for
    for (int64_t i = 0; i < length; i++) {
        int64_t temp = quantFunc(inputData[i]);
        temp = temp < clipMin ? clipMin : temp;
        temp = temp > clipMax ? clipMax : temp;
        outputData[i] = static_cast<float>(temp - calParams.offset);
    }
    return AmctCommon::SUCCESS;
}


template<class T>
Status FakeQuantKernelOutputInt8(const T* inputData, 
                                 int8_t* outputData, 
                                 int64_t length, 
                                 FakeCalParams calParams)
{
    int64_t clipMin = -pow(BINARY_BASE, QUANT_BIT_NUM - 1);
    int64_t clipMax = pow(BINARY_BASE, QUANT_BIT_NUM - 1) - 1;

    auto quantFunc = [calParams](T x) {
        if (calParams.fakePrecisonMode == util::FORCE_FP16_QUANT) {
            return rint(util::CastToFP16PrecisionCPU(util::CastToFP16PrecisionCPU(
                (util::CastToFP16PrecisionCPU(x) * util::FakeFp16PrecisionDataCPU(calParams.scale))) + calParams.offset));
        } else {
            return rint(x * calParams.scale) + calParams.offset;
        }
    };
    // Do quant computation
#pragma omp parallel for
    for (int64_t i = 0; i < length; i++) {
        int64_t temp = quantFunc(inputData[i]);
        temp = temp < clipMin ? clipMin : temp;
        temp = temp > clipMax ? clipMax : temp;
        outputData[i] = static_cast<int8_t>(temp);
    }
    return AmctCommon::SUCCESS;
}


template<class T>
Status FakeAntiQuantKernel(const T* inputData, T* outputData, int64_t length, float scale)
{
#pragma omp parallel for
    for (int64_t idx = 0; idx < length; idx++) {
        outputData[idx] = inputData[idx] * scale;
    }
    return AmctCommon::SUCCESS;
}


int ParseParamData(
    DequantParam& dequantParam)
{
    const uint64_t shiftnMask = 0x000000ff00000000;
    const uint64_t deqscaleMask = 0x00000000ffffffff;
#pragma omp parallel for
    for (int64_t idx = 0; idx < dequantParam.paramSize; idx++) {
        unsigned int shiftValue = (dequantParam.paramData[idx] & shiftnMask) >> SHIFT_VAL_LEN;
        dequantParam.shiftValue[idx] = pow(BINARY_BASE, shiftValue);
        if (shiftValue != 0) {
            dequantParam.clipMode = CLIP_16;
        }
        unsigned int deqscaleUint = dequantParam.paramData[idx] & deqscaleMask;
        float* deqscalePtr = reinterpret_cast<float*>(&deqscaleUint);
        dequantParam.deqScale[idx] = deqscalePtr[0];
    }
    return AmctCommon::SUCCESS;
}


int FakeDequant(InputDataParam param,
                DequantParam dequantParam)
{
    int res = AmctCommon::SUCCESS;
    if (param.outType != FLOAT_TYPE_ID) {
        std::vector<float> outCast(param.length, 0);
        // in_16, out_16
        if (param.inType != FLOAT_TYPE_ID) {
            std::vector<float> inCast(param.length, 0);
            DataCastToFloat32Functor<util::CPUDevice, uint16_t>()(
                reinterpret_cast<const uint16_t*>(param.in), inCast.data(), param.length);
            res = FakeDequantKernel(inCast.data(), outCast.data(), param.length, dequantParam, param.fakePrecisionMode);
        } else {
            // in_32, out_16
            res = FakeDequantKernel(reinterpret_cast<const float*>(param.in), outCast.data(),
                param.length, dequantParam, param.fakePrecisionMode);
        }
        DataCastToFloat16Functor<util::CPUDevice, float>()(
            outCast.data(), reinterpret_cast<uint16_t*>(param.out), param.length);
        return res;
    }
    // in_32, out_32
    res = FakeDequantKernel(reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
        param.length, dequantParam, param.fakePrecisionMode);
    return res;
}


int FakeQuant(InputDataParam param, int64_t quantBits, float scale, int64_t offset)
{
    FakeCalParams calParams = {param.fakePrecisionMode, scale, offset};
    if (param.inType == FLOAT16_TYPE_ID) {
        std::vector<float> inCast(param.length, 0);
        DataCastToFloat32Functor<util::CPUDevice, uint16_t>()(
            reinterpret_cast<const uint16_t*>(param.in), inCast.data(), param.length);
        // in_16, out_16
        if (param.outType == FLOAT16_TYPE_ID) {
            std::vector<float> outCast(param.length, 0);
            int res = FakeQuantKernel(inCast.data(), outCast.data(), param.length, quantBits, calParams);
            DataCastToFloat16Functor<util::CPUDevice, float>()(
                outCast.data(), reinterpret_cast<uint16_t*>(param.out), param.length);
            return res;
        }

        // in_16, out_32
        if (param.outType == FLOAT_TYPE_ID) {
            return FakeQuantKernel(inCast.data(), reinterpret_cast<float*>(param.out), param.length,
                quantBits, calParams);
        }

        // in_16, out_8
        if (param.outType == INT8_TYPE_ID) {
            return FakeQuantKernelOutputInt8(inCast.data(), reinterpret_cast<int8_t*>(param.out), param.length,
                calParams);
        }
    }

    // in_32, out_8
    if (param.outType == INT8_TYPE_ID) {
        return FakeQuantKernelOutputInt8(reinterpret_cast<const float*>(param.in), reinterpret_cast<int8_t*>(param.out),
            param.length, calParams);
    }

    // in_32, out_32
    return FakeQuantKernel(reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
        param.length, quantBits, calParams);
}


int FakeAntiQuant(InputDataParam param, float scaleData)
{
    if (param.outType != FLOAT_TYPE_ID) {
        std::vector<float> outCast(param.length, 0);
        // in_16, out_16
        if (param.inType != FLOAT_TYPE_ID) {
            std::vector<float> inCast(param.length, 0);
            DataCastToFloat32Functor<util::CPUDevice, uint16_t>()(
                reinterpret_cast<const uint16_t*>(param.in), inCast.data(), param.length);
            FakeAntiQuantKernel(inCast.data(), outCast.data(), param.length, scaleData);
        } else {
            // in_32, out_16
            FakeAntiQuantKernel(reinterpret_cast<const float*>(param.in), outCast.data(), param.length, scaleData);
        }
        DataCastToFloat16Functor<util::CPUDevice, float>()(outCast.data(), reinterpret_cast<uint16_t*>(param.out),
                                                           param.length);
        return AmctCommon::SUCCESS;
    }
    // in_32, out_32
    FakeAntiQuantKernel(reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
                        param.length, scaleData);
    return AmctCommon::SUCCESS;
}