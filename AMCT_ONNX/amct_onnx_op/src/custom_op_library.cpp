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

#include <mutex>
#include <cmath>

#include "dequant_quant.h"
#include "amct_utils.h"
#include "ifmr_kernel.h"
#include "search_n_kernel.h"
#include "search_n_v2_kernel.h"
#include "quant_kernel.h"
#include "dequant_kernel.h"
#include "ascend_quant_kernel.h"
#include "ascend_dequant_kernel.h"
#include "ascend_antiquant_kernel.h"
#include "dmq_balance_kernel.h"
#include "hfmg_kernel.h"
#include "search_n.h"
#include "util.h"
#include "dump_kernel.h"


struct OrtCustomOpDomainDeleter {
    explicit OrtCustomOpDomainDeleter(const OrtApi* ortApi) : ortApi_(ortApi)
    {
    }
    void operator()(OrtCustomOpDomain* domain) const
    {
        ortApi_->ReleaseCustomOpDomain(domain);
    }
    const OrtApi* ortApi_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> g_ortCustomOpDomainContainer;
static std::mutex g_ortCustomOpDomainMutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ortApi)
{
    std::lock_guard<std::mutex> lock(g_ortCustomOpDomainMutex);
    auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(
        domain, OrtCustomOpDomainDeleter(ortApi));
    g_ortCustomOpDomainContainer.push_back(std::move(ptr));
}


struct IFMROp : Ort::CustomOpBase<IFMROp, IFMRKernel> {
public:
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new IFMRKernel(api, info);
    }
    const char* GetName() const
    {
        return "IFMR";
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    size_t GetInputTypeCount() const
    {
        return 1;
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
} g_cIFMROp;


struct HFMGOp : Ort::CustomOpBase<HFMGOp, HFMGKernel> {
public:
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new HFMGKernel(api, info);
    }
    const char* GetName() const
    {
        return "HFMG";
    }
    size_t GetInputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
} g_cHFMGOp;


struct DUMPOp : Ort::CustomOpBase<DUMPOp, DUMPKernel> {
public:
    size_t GetInputTypeCount() const
    {
        return 1;
    }
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new DUMPKernel(api, info);
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    const char* GetName() const
    {
        return "DUMP";
    }
    size_t GetOutputTypeCount() const
    {
        return 0;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
} g_cDUMPOp;


struct QuantOp : Ort::CustomOpBase<QuantOp, QuantKernel> {
public:
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new QuantKernel(api, info);
    }
    const char* GetName() const
    {
        return "Quant";
    }
    size_t GetInputTypeCount() const
    {
        return 1;
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
} g_cQuantOp;


struct DequantOp : Ort::CustomOpBase<DequantOp, DequantKernel> {
public:
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new DequantKernel(api, info);
    }
    size_t GetInputTypeCount() const
    {
        return NUM_THREE;
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    const char* GetName() const
    {
        return "Dequant";
    }
} g_cDequantOp;


struct AscendQuantOp : Ort::CustomOpBase<AscendQuantOp,  AscendQuantKernel> {
public:
    explicit AscendQuantOp(const char* provider, void* compute_stream) : provider_(provider),
        compute_stream_(compute_stream) {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new AscendQuantKernel(api, info);
    }
    const char* GetExecutionProviderType() const
    {
        return provider_;
    }
    size_t GetInputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    const char* GetName() const
    {
        return "AscendQuant";
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    virtual ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
private:
    const char* provider_;
    void* compute_stream_;
};


#if USE_CUDA
    AscendQuantOp g_cAscendQuantOp{"CUDAExecutionProvider", nullptr};
#else
    AscendQuantOp g_cAscendQuantOp{"CPUExecutionProvider", nullptr};
#endif


struct AscendDequantOp : Ort::CustomOpBase<AscendDequantOp, AscendDequantKernel> {
public:
    explicit AscendDequantOp(const char* provider, void* compute_stream) : provider_(provider),
        compute_stream_(compute_stream) {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new AscendDequantKernel(api, info);
    }
    const char* GetName() const
    {
        return "AscendDequant";
    }
    const char* GetExecutionProviderType() const
    {
        return provider_;
    }
    size_t GetInputTypeCount() const
    {
        return NUM_TWO;
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    virtual ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    ONNXTensorElementDataType GetInputType(size_t index) const
    {
        if (index == 0) {
            return AmctUtils::AmctOpDynamicTypeCheck();
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }
private:
    const char* provider_;
    void* compute_stream_;
};

#if USE_CUDA
    AscendDequantOp g_cAscendDequantOp{"CUDAExecutionProvider", nullptr};
#else
    AscendDequantOp g_cAscendDequantOp{"CPUExecutionProvider", nullptr};
#endif


struct AscendAntiQuantOp : Ort::CustomOpBase<AscendAntiQuantOp, AntiQuantKernel> {
public:
    explicit AscendAntiQuantOp(const char* provider, void* compute_stream) : provider_(provider),
        compute_stream_(compute_stream) {}
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new AntiQuantKernel(api, info);
    }
    const char* GetName() const
    {
        return "AscendAntiQuant";
    }
    const char* GetExecutionProviderType() const
    {
        return provider_;
    }
    size_t GetOutputTypeCount() const
    {
        return 1;
    }
    virtual ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }
    size_t GetInputTypeCount() const
    {
        return 1;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }

private:
    const char* provider_;
    void* compute_stream_;
};

#if USE_CUDA
    AscendAntiQuantOp g_cAscendAntiQuantOp{"CUDAExecutionProvider", nullptr};
#else
    AscendAntiQuantOp g_cAscendAntiQuantOp{"CPUExecutionProvider", nullptr};
#endif


struct SearchNOp : Ort::CustomOpBase<SearchNOp, SearchNKernel> {
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new SearchNKernel(api, info);
    }
    const char* GetName() const
    {
        return "SearchN";
    }
    size_t GetInputTypeCount() const
    {
        return SEARCHN_INPUT_TYPE_COUNT;
    }
    ONNXTensorElementDataType GetInputType(size_t index) const
    {
        if (index == 0) {
            return AmctUtils::AmctOpDynamicTypeCheck();
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const
    {
        return 0;
    }
} c_SearchNOp;


struct SearchNV2Op : Ort::CustomOpBase<SearchNV2Op, SearchNV2Kernel> {
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new SearchNV2Kernel(api, info);
    }
    const char* GetName() const
    {
        return "SearchNv2";
    }
    size_t GetInputTypeCount() const
    {
        return SEARCHNV2_INPUT_TYPE_COUNT;
    }
    ONNXTensorElementDataType GetInputType(size_t index) const
    {
        if (index == 0) {
            return AmctUtils::AmctOpDynamicTypeCheck();
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const
    {
        return 0;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
} c_SearchNV2Op;


struct DMQBalanceOp : Ort::CustomOpBase<DMQBalanceOp, DMQBalanceKernel> {
public:
#if ORT_API_VERSION >= 16
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void* kernel) const
    {
        kernel = CreateKernel(api, info);
        return api.CreateStatus(ORT_OK, "Success");
    }
#endif
    explicit DMQBalanceOp(const char* provider, void* compute_stream) : provider_(provider),
        compute_stream_(compute_stream) {}
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new DMQBalanceKernel(api, info);
    }
    const char* GetName() const
    {
        return "DMQBalancer";
    }
    const char* GetExecutionProviderType() const
    {
        return provider_;
    }
    size_t GetOutputTypeCount() const
    {
        return 0;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetInputTypeCount() const
    {
        return DMQB_INPUT_TYPE_COUNT;
    }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return AmctUtils::AmctOpDynamicTypeCheck();
    }

private:
    const char* provider_;
    void* compute_stream_;
};

#if USE_CUDA
    DMQBalanceOp g_cDMQBalanceOp{"CUDAExecutionProvider", nullptr};
#else
    DMQBalanceOp g_cDMQBalanceOp{"CPUExecutionProvider", nullptr};
#endif


struct AscendQuantOpFp16 : AscendQuantOp {
public:
    explicit AscendQuantOpFp16(const char* provider, void* compute_stream)
        : AscendQuantOp(provider, compute_stream), provider_(provider), compute_stream_(compute_stream) {}
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
private:
    const char* provider_;
    void* compute_stream_;
};


#if USE_CUDA
    AscendQuantOpFp16 g_cAscendQuantOpFp16{"CUDAExecutionProvider", nullptr};
#else
    AscendQuantOpFp16 g_cAscendQuantOpFp16{"CPUExecutionProvider", nullptr};
#endif


struct AscendQuantOpInt8 : AscendQuantOp {
public:
    explicit AscendQuantOpInt8(const char* provider, void* compute_stream)
        : AscendQuantOp(provider, compute_stream), provider_(provider), compute_stream_(compute_stream) {}
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }
private:
    const char* provider_;
    void* compute_stream_;
};


#if USE_CUDA
    AscendQuantOpInt8 g_cAscendQuantOpInt8{"CUDAExecutionProvider", nullptr};
#else
    AscendQuantOpInt8 g_cAscendQuantOpInt8{"CPUExecutionProvider", nullptr};
#endif


struct AscendDequantOpFp16 : AscendDequantOp {
public:
    explicit AscendDequantOpFp16(const char* provider, void* compute_stream)
        : AscendDequantOp(provider, compute_stream), provider_(provider), compute_stream_(compute_stream) {}
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
private:
    const char* provider_;
    void* compute_stream_;
};

#if USE_CUDA
    AscendDequantOpFp16 g_cAscendDequantOpFp16{"CUDAExecutionProvider", nullptr};
#else
    AscendDequantOpFp16 g_cAscendDequantOpFp16{"CPUExecutionProvider", nullptr};
#endif


struct AscendAntiQuantOpFp16 : AscendAntiQuantOp {
public:
    explicit AscendAntiQuantOpFp16(const char* provider, void* compute_stream)
        : AscendAntiQuantOp(provider, compute_stream), provider_(provider), compute_stream_(compute_stream) {}
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
private:
    const char* provider_;
    void* compute_stream_;
};

#if USE_CUDA
    AscendAntiQuantOpFp16 g_cAscendAntiQuantOpFp16{"CUDAExecutionProvider", nullptr};
#else
    AscendAntiQuantOpFp16 g_cAscendAntiQuantOpFp16{"CPUExecutionProvider", nullptr};
#endif

static OrtStatus* RegisterCustomInt8Domain(OrtSessionOptions* options, const OrtApi* ortApi)
{
    // register customop int8 domain
    const char* cOpDomainInt8 = "amct.customop.extint8";
    OrtCustomOpDomain* domainExInt8 = nullptr;
    if (auto status = ortApi->CreateCustomOpDomain(cOpDomainInt8, &domainExInt8)) {
        return status;
    }
    AddOrtCustomOpDomainToContainer(domainExInt8, ortApi);
    // add ascendquant int8 custom op
    if (auto status = ortApi->CustomOpDomain_Add(domainExInt8, &g_cAscendQuantOpInt8)) {
        return status;
    }
    return ortApi->AddCustomOpDomain(options, domainExInt8);
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
{
    // register customop domain
    const char* cOpDomain = "amct.customop";
    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);
    if (auto status = ortApi->CreateCustomOpDomain(cOpDomain, &domain)) {
        return status;
    }
    AddOrtCustomOpDomainToContainer(domain, ortApi);
    // add IFMR custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cIFMROp)) {
        return status;
    }
    // add HFMG custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cHFMGOp)) {
        return status;
    }
    // add dump custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cDUMPOp)) {
        return status;
    }
    // add quant custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cQuantOp)) {
        return status;
    }
    // add dequant custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cDequantOp)) {
        return status;
    }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cAscendQuantOp)) {
        return status;
    }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cAscendDequantOp)) {
        return status;
    }
    // add antiquant custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cAscendAntiQuantOp)) {
        return status;
    }
    // add search_n custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SearchNOp)) {
        return status;
    }
    // add search_n_v2 custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SearchNV2Op)) {
        return status;
    }
    // add dmq_balance custom op
    if (auto status = ortApi->CustomOpDomain_Add(domain, &g_cDMQBalanceOp)) {
        return status;
    }
    if (auto status = ortApi->AddCustomOpDomain(options, domain)) {
        return status;
    }

    // register customop domain
    const char* cOpDomainEx = "amct.customop.extfp16";
    OrtCustomOpDomain* domainEx = nullptr;
    if (auto status = ortApi->CreateCustomOpDomain(cOpDomainEx, &domainEx)) {
        return status;
    }
    AddOrtCustomOpDomainToContainer(domainEx, ortApi);
    // add ascendquant fp16 custom op
    if (auto status = ortApi->CustomOpDomain_Add(domainEx, &g_cAscendQuantOpFp16)) {
        return status;
    }
    // add ascenddequant fp16 custom op
    if (auto status = ortApi->CustomOpDomain_Add(domainEx, &g_cAscendDequantOpFp16)) {
        return status;
    }
    // add ascendantiquant fp16 custom op
    if (auto status = ortApi->CustomOpDomain_Add(domainEx, &g_cAscendAntiQuantOpFp16)) {
        return status;
    }
    if (auto status = ortApi->AddCustomOpDomain(options, domainEx)) {
        return status;
    }

    return RegisterCustomInt8Domain(options, ortApi);
}
