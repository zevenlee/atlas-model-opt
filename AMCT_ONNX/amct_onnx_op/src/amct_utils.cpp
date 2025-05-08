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
 * @brief amct_utils C++ implementation
 * @file amct_utils.cpp
 *
 * @version 1.0
 */

#include "amct_utils.h"
#include "util.h"
#include "cast_util.h"

namespace AmctUtils {
    void AmctDumpData(const char* filePath,
                      const int32_t* inputShapeArray,
                      int shapeLen,
                      const void* inputDataArray,
                      int dataLen)
    {
        std::ofstream outShape(filePath, std::ios::binary);
        CHECK_TRUE_RETURN_WITH_LOG(!outShape.is_open(), "AmctDumpData write outShape fail to open file.\n");
        outShape.write(reinterpret_cast<const char*>(inputShapeArray), sizeof(int32_t) * shapeLen);
        outShape.close();
        std::ofstream outData(filePath, std::ios::binary | std::ios::app);
        CHECK_TRUE_RETURN_WITH_LOG(!outData.is_open(), "AmctDumpData write outData fail to open file.\n");
        outData.write(reinterpret_cast<const char*>(inputDataArray), dataLen);
        outData.close();
    }

    void ConvertLayerName(std::string& originalLayerName,
                          const std::string& subString,
                          const std::string& replaceString)
    {
        std::string::size_type pos = 0;
        std::string::size_type subStrLength = subString.size();
        std::string::size_type replaceStringLen = replaceString.size();
        while ((pos = originalLayerName.find(subString, pos)) != std::string::npos) {
            originalLayerName.replace(pos, subStrLength, replaceString);
            pos += replaceStringLen;
        }
    }
    std::string TrimTailSpace(std::string& str)
    {
        std::string::size_type pos = str.find_last_not_of(' ');
        std::string trimedStr;
        if (pos != std::string::npos) {
            trimedStr = str.substr(0, pos + 1);
        } else {
            trimedStr = str;
        }
        return trimedStr;
    }

    void CheckTensorNotEmpty(size_t tensorElementSize)
    {
        if (tensorElementSize == 0) {
            ORT_CXX_API_THROW("AMCT can't not support empty tensor, please check model input.", ORT_FAIL);
        }
    }

    ONNXTensorElementDataType AmctOpDynamicTypeCheck()
    {
        if (ORT_API_VERSION < SUPPORT_DYN_TYPE_ORT_VERSION) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    void SaveInputDataToFloat32(const void* inputData, float* saveData, size_t dataLength, int dataId)
    {
        const float* dataIn = nullptr;
        if (dataId == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            dataIn = reinterpret_cast<const float*>(inputData);
            for (size_t i = 0; i < dataLength; i++) {
                saveData[i] = dataIn[i];
            }
            return;
        } else if (dataId == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            auto castIn = reinterpret_cast<const uint16_t*>(inputData);
            util::DataCastToFloat32Functor<util::CPUDevice, uint16_t>()(castIn, saveData, dataLength);
            return;
        } else {
            ORT_CXX_API_THROW("AMCT cannot accept types other than float and float16.", ORT_FAIL);
        }
    }
    std::string GetStringAttr(const OrtApi& api, const OrtKernelInfo* info, const std::string& attrName)
    {
#if ORT_API_VERSION >= 16
        size_t size = 0;
        CheckStatus(api, api.KernelInfoGetAttribute_string(info, attrName.c_str(), nullptr, &size));
        std::string attrValue;
        attrValue.resize(size);
        CheckStatus(api, api.KernelInfoGetAttribute_string(info, attrName.c_str(), &attrValue[0], &size));
        attrValue.resize(size - 1);
        return attrValue;
#else
        Ort::CustomOpApi customOpApi(api);
        return customOpApi.KernelInfoGetAttribute<std::string>(info, attrName.c_str());
#endif
    }
}
