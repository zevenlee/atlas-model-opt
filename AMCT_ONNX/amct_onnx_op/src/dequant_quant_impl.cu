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
* @brief fake_dequant_quant custom op kernel func CUDA implement
*
* @file dequant_quant_impl.cpp
*
* @version 1.0
*/
#include "dequant_quant.h"
#include <cmath>
#include <mutex>
#include <cfloat>
#include <algorithm>
#include <memory>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "util.h"
#include "cast_util.h"
#include "cast.cuh"

using namespace util;
#define FLOAT_TYPE_ID (1)
#define INT8_TYPE_ID (3)
#define FLOAT16_TYPE_ID (10)

__global__ void FakeQuantCudaImpl(
    const float* inputData,
    float* outputData,
    int64_t length,
    int64_t bound,
    FakeCalParams calParams)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataTmp = 0;
    if (tid < length) {
        if (calParams.fakePrecisonMode == util::FORCE_FP16_QUANT) {
            dataTmp = static_cast<int>(rint(CastToFP16Precision(
                CastToFP16Precision(CastToFP16Precision(inputData[tid]) * FakeFp16PrecisionData(calParams.scale)) + calParams.offset)));
        } else {
            dataTmp = static_cast<int>(rint(inputData[tid] * calParams.scale)) + calParams.offset;
        }
        dataTmp = (dataTmp < -bound) ? -bound : dataTmp;
        dataTmp = (dataTmp > bound - 1) ? bound - 1 : dataTmp;
        outputData[tid] = static_cast<float>((dataTmp - calParams.offset));
    }
}


__global__ void FakeQuantInt8CudaImpl(
    const float* inputData,
    int8_t* outputData,
    int64_t length,
    int64_t bound,
    FakeCalParams calParams)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataTmp = 0;
    if (tid < length) {
        if (calParams.fakePrecisonMode == util::FORCE_FP16_QUANT) {
            if ((calParams.scale > MAX_FP16) || (calParams.scale > 0 && calParams.scale < MIN_FP16)) {
                calParams.scale = CastToFP16Precision(sqrt(calParams.scale)) * CastToFP16Precision(sqrt(calParams.scale));
            } else {
                calParams.scale = CastToFP16Precision(calParams.scale);
            }
            dataTmp = static_cast<int>(rint(CastToFP16Precision(CastToFP16Precision(inputData[tid]) * calParams.scale))) 
                + calParams.offset;
        } else {
            dataTmp = static_cast<int>(rint(inputData[tid] * calParams.scale)) + calParams.offset;
        }
        dataTmp = (dataTmp < -bound) ? -bound : dataTmp;
        dataTmp = (dataTmp > bound - 1) ? bound - 1 : dataTmp;
        outputData[tid] = static_cast<float>(dataTmp);
    }
}


__global__ void FakeDeQuantCudaImpl(
    const float* inputData,
    float* outputData,
    int64_t length,
    struct DequantParam dequantParam,
    int64_t fakePrecisionMode)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        int channelIndex = !dequantParam.channelWise ? 0 : (tid % (dequantParam.chwSize)) / dequantParam.hwSize;
        unsigned int shiftBit = (dequantParam.paramData[channelIndex] & dequantParam.shiftnMask) >> 32;
        if (shiftBit != 0 ) {
            dequantParam.clipMode = CLIP_16;
        }
        float clipMin = -static_cast<float>(pow(BINARY_BASE, dequantParam.clipMode - 1));
        float clipMax = static_cast<float>(pow(BINARY_BASE, dequantParam.clipMode - 1) - 1);
        float shiftValue = pow(BINARY_BASE, shiftBit);
        unsigned int deqscaleUint = dequantParam.paramData[channelIndex] & dequantParam.deqscaleMask;
        float* deqscalePtr = reinterpret_cast<float*>(&deqscaleUint);

        float tmpData = 0;
        if (std::fabs(shiftValue - 1) <= FLT_EPSILON) {
            tmpData = inputData[tid];
        } else {
            tmpData = floor(inputData[tid] / shiftValue);
        }
        tmpData = (tmpData < clipMin) ? clipMin : tmpData;
        tmpData = (tmpData > clipMax) ? clipMax : tmpData;
        if (fakePrecisionMode == util::FORCE_FP16_QUANT) {
            outputData[tid] = CastToFP16Precision(tmpData * CastToS19(*deqscalePtr) * shiftValue);
        } else {
            outputData[tid] = tmpData * (*deqscalePtr) * shiftValue;
        }
    }
}


__global__ void FakeAntiQuantCudaImpl(
    const float* inputData,
    float* outputData,
    int64_t length,
    float scaleData)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        outputData[tid] = inputData[tid] * scaleData;
    }
}

inline int checkCudaError(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
        return AmctCommon::CUDA_ASYNC_ERROR;
    }
    return AmctCommon::SUCCESS;
}

int FakeQuantCuda(InputDataParam param,
    int64_t quantBits,
    float scale,
    int64_t offset)
{
    FakeCalParams calParams = {param.fakePrecisionMode, scale, offset};
    int64_t bound = pow(2, quantBits - 1);
    cudaError_t errAsync = cudaDeviceSynchronize();
    checkCudaError(errAsync, __FILE__, __LINE__);

    thrust::device_vector<float> inCast;
    if (param.inType == FLOAT16_TYPE_ID) {
        inCast.resize(param.length);
        float* inSavePtr = thrust::raw_pointer_cast(inCast.data());
        DataCastToFloat32Functor<util::GPUDevice, uint16_t>()(reinterpret_cast<const uint16_t*>(param.in), inSavePtr,
            param.length);
        if (param.outType == INT8_TYPE_ID) {
            // in_16, out_8
            FakeQuantInt8CudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
                inSavePtr, reinterpret_cast<int8_t*>(param.out), param.length, bound, calParams);
        } else if (param.outType == FLOAT16_TYPE_ID) {
            // in_16, out_16
            thrust::device_vector<float> outCast(param.length, 0);
            float* outSavePtr = thrust::raw_pointer_cast(outCast.data());
            FakeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
                inSavePtr, outSavePtr, param.length, bound, calParams);
            DataCastToFloat16Functor<util::GPUDevice, float>()(outSavePtr, reinterpret_cast<uint16_t*>(param.out),
                param.length);
        } else {
            // in_16, out_32
            FakeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
                inSavePtr, reinterpret_cast<float*>(param.out), param.length, bound, calParams);
        }
        errAsync = cudaDeviceSynchronize();
        checkCudaError(errAsync, __FILE__, __LINE__);
        if (errAsync != cudaSuccess) {
            return AmctCommon::CUDA_ERROR;
        }
        return AmctCommon::SUCCESS;
    }

    if (param.outType == INT8_TYPE_ID) {
        // in_32, out_8
        FakeQuantInt8CudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
            reinterpret_cast<const float*>(param.in), reinterpret_cast<int8_t*>(param.out),
            param.length, bound, calParams);
    } else {
        // in_32, out_32
        FakeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
            reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
            param.length, bound, calParams);
    }

    errAsync = cudaDeviceSynchronize();
    checkCudaError(errAsync, __FILE__, __LINE__);
    if (errAsync != cudaSuccess) {
        return AmctCommon::CUDA_ERROR;
    }
    return AmctCommon::SUCCESS;
}


int FakeDequantCuda(InputDataParam param, DequantParam dequantParam)
{
    dequantParam.shiftnMask = 0x000000ff00000000;
    dequantParam.deqscaleMask = 0x00000000ffffffff;
    cudaError_t errAsync = cudaDeviceSynchronize();
    checkCudaError(errAsync, __FILE__, __LINE__);
    if (errAsync != cudaSuccess) {
        return AmctCommon::CUDA_ERROR;
    }
    if (param.outType != 1) {
        thrust::device_vector<float> outCast(param.length, 0);
        float* outSavePtr = thrust::raw_pointer_cast(outCast.data());
        // in_16, out_16
        if (param.inType != 1) {
            thrust::device_vector<float> inCast(param.length, 0);
            float* inSavePtr = thrust::raw_pointer_cast(inCast.data());
            DataCastToFloat32Functor<util::GPUDevice, uint16_t>()(reinterpret_cast<const uint16_t*>(param.in), inSavePtr, param.length);
            FakeDeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS, 0, 0>>>(
                inSavePtr, outSavePtr, param.length, dequantParam, param.fakePrecisionMode);
        } else {
            // in_32, out_16
            FakeDeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS, 0, 0>>>(
                reinterpret_cast<const float*>(param.in), outSavePtr, param.length, dequantParam, param.fakePrecisionMode);
        }
        DataCastToFloat16Functor<util::GPUDevice, float>()(outSavePtr, reinterpret_cast<uint16_t*>(param.out), param.length);
        errAsync = cudaDeviceSynchronize();
        checkCudaError(errAsync, __FILE__, __LINE__);
        if (errAsync != cudaSuccess) {
            return AmctCommon::CUDA_ERROR;
        }
        return AmctCommon::SUCCESS;
    }
    // in_32, out_32
    FakeDeQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS, 0, 0>>>(
        reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
        param.length, dequantParam, param.fakePrecisionMode);
    errAsync = cudaDeviceSynchronize();
    checkCudaError(errAsync, __FILE__, __LINE__);
    if (errAsync != cudaSuccess) {
        return AmctCommon::CUDA_ERROR;
    }
    return AmctCommon::SUCCESS;
}


int FakeAntiQuantCuda(InputDataParam param, float scaleData)
{
    cudaError_t errAsync;
    if (param.outType != 1) {
        thrust::device_vector<float> outCast(param.length, 0);
        float* outSavePtr = thrust::raw_pointer_cast(outCast.data());
        // in_16, out_16
        if (param.inType != 1) {
            thrust::device_vector<float> inCast(param.length, 0);
            float* inSavePtr = thrust::raw_pointer_cast(inCast.data());
            DataCastToFloat32Functor<util::GPUDevice, uint16_t>()(reinterpret_cast<const uint16_t*>(param.in), inSavePtr, param.length);
            FakeAntiQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
                inSavePtr, outSavePtr, param.length, scaleData);
        } else {
            // in_32, out_16
            FakeAntiQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
                reinterpret_cast<const float*>(param.in), outSavePtr, param.length, scaleData);
        }
        DataCastToFloat16Functor<util::GPUDevice, float>()(outSavePtr, reinterpret_cast<uint16_t*>(param.out), param.length);
        errAsync = cudaDeviceSynchronize();
        checkCudaError(errAsync, __FILE__, __LINE__);
        if (errAsync != cudaSuccess) {
            return AmctCommon::CUDA_ERROR;
        }
        return AmctCommon::SUCCESS;
    }
    // in_32, out_32
    FakeAntiQuantCudaImpl<<<AMCT_GET_BLOCKS(param.length), AMCT_CUDA_NUM_THREADS>>>(
        reinterpret_cast<const float*>(param.in), reinterpret_cast<float*>(param.out),
        param.length, scaleData);
    errAsync = cudaDeviceSynchronize();
    checkCudaError(errAsync, __FILE__, __LINE__);
    if (errAsync != cudaSuccess) {
        return AmctCommon::CUDA_ERROR;
    }
    return AmctCommon::SUCCESS;
}