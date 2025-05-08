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
 * @brief dequant_quant head file
 *
 * @file dequant_quant.h
 *
 * @version 1.0
 */

#ifndef DEQUANT_QUANT_H
#define DEQUANT_QUANT_H
#include <cstdint>
#include <string>

#ifdef __cplusplus
extern "C"
{
#endif

const int CLIP_32 = 32;
const int CLIP_16 = 16;

struct DequantParam {
    int64_t chwSize;
    int64_t hwSize;
    int64_t clipMode;
    uint64_t shiftnMask;
    uint64_t deqscaleMask;
    int64_t paramSize;
    const uint64_t* paramData;
    float* shiftValue;
    float* deqScale;
    float* shiftValueDevice;
    float* deqScaleDevice;
    bool channelWise;
};

struct InputDataParam {
    const void* in;
    void* out;
    int64_t inType;
    int64_t outType;
    size_t length;
    int64_t fakePrecisionMode;
};

struct FakeCalParams {
    int64_t fakePrecisonMode;
    float scale;
    int64_t offset;
}; 

int FakeDequant(InputDataParam inputDataParam,
                DequantParam dequantParam);

int FakeQuant(InputDataParam inputDataParam,
              int64_t quantBits,
              float scale,
              int64_t offset);

int FakeQuantCuda(InputDataParam inputDataParam,
                  int64_t quantBits,
                  float scale,
                  int64_t offset);

int FakeDequantCuda(InputDataParam inputDataParam,
                    DequantParam dequantParam);

int FakeAntiQuantCuda(InputDataParam inputDataParam,
                      float scaleData);

int FakeAntiQuant(InputDataParam param,
                  float scaleData);

int ParseParamData(DequantParam& dequantParam);

int ParseParamDataCuda(DequantParam& dequantParam);


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* DEQUANT_QUANT_H */
