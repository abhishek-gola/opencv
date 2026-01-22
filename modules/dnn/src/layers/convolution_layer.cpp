/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <numeric>

#include "../opdata_factory.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/convolution.hpp"
#include "../cuda4dnn/primitives/transpose_convolution.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

#include "cpu_kernels/convolution.hpp"

namespace cv
{
namespace dnn
{

namespace {

// Data-only op descriptor (ENGINE_NEW): ConvLayerData : LayerOpData
class ConvLayerData CV_FINAL : public LayerOpData
{
public:
    std::vector<size_t> kernel_size, pads_begin, pads_end, strides, dilations, adjust_pads;
    String padMode;
    bool useWinograd = false;
    int groups = 1;

    void initFromParams()
    {
        groups = params.get<int>("group", 1);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations,
                                   padMode, adjust_pads, useWinograd);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        std::cout<<"ConvLayerData::getMemoryShapes"<<std::endl;
        CV_Assert(requiredOutputs == 0 || requiredOutputs == 1);
        CV_Assert(!inputs.empty());
        CV_Assert(inputs[0].size() > 2);

        // weights may come either from blobs (params.blobs) or as the 2nd input
        const bool haveBlobs = !params.blobs.empty();
        CV_Assert(haveBlobs || inputs.size() > 1);
        const int* weightShape = haveBlobs ? params.blobs[0].size.p : &inputs[1][0];

        internals.clear();

        std::vector<int> inpShape(inputs[0].begin() + 2, inputs[0].end());
        int outCn = weightShape[0];

        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);  // batch
        outShape.push_back(outCn);

        if (padMode.empty())
        {
            for (int i = 0; i < (int)inpShape.size(); i++)
            {
                outShape.push_back((inpShape[i] + (int)pads_begin[i] + (int)pads_end[i] -
                                    (int)dilations[i] * ((int)kernel_size[i] - 1) - 1) / (int)strides[i] + 1);
            }
        }
        else
        {
            getConvPoolOutParams(inpShape, kernel_size, strides, padMode, dilations, outShape);
        }

        outputs.resize(1, MatShape(outShape));
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        // Convolution internals (if any) are float32 buffers today.
        internals.assign(requiredInternals, CV_32F);
    }

    std::ostream& dump(std::ostream& strm, int indent, bool comma) const CV_OVERRIDE
    {
        auto prindent_ = [&strm](int n) -> std::ostream& {
            for (int i = 0; i < n; ++i) strm << ' ';
            return strm;
        };
        auto dumpVec_ = [&strm](const std::vector<size_t>& v) -> std::ostream& {
            strm << '[';
            for (size_t i = 0; i < v.size(); ++i)
            {
                if (i) strm << ", ";
                strm << v[i];
            }
            strm << ']';
            return strm;
        };

        int subindent = indent + 3;
        prindent_(indent); strm << "{\n";
        prindent_(subindent); strm << "name: \"" << name << "\",\n";
        prindent_(subindent); strm << "type: \"" << type << "\",\n";
        prindent_(subindent); strm << "groups: " << groups << ",\n";
        prindent_(subindent); strm << "kernel_size: "; dumpVec_(kernel_size); strm << ",\n";
        prindent_(subindent); strm << "strides: "; dumpVec_(strides); strm << ",\n";
        prindent_(subindent); strm << "dilations: "; dumpVec_(dilations); strm << ",\n";
        prindent_(subindent); strm << "pads_begin: "; dumpVec_(pads_begin); strm << ",\n";
        prindent_(subindent); strm << "pads_end: "; dumpVec_(pads_end); strm << ",\n";
        prindent_(subindent); strm << "padMode: \"" << padMode << "\",\n";
        prindent_(subindent); strm << "useWinograd: " << (useWinograd ? "true" : "false") << "\n";
        prindent_(indent); strm << "}";
        if (comma) strm << ",";
        strm << "\n";
        return strm;
    }
};

static Ptr<LayerOpData> createConvLayerData(const LayerParams& lp,
                                           const std::vector<Arg>& inputs,
                                           const std::vector<Arg>& outputs)
{
    Ptr<ConvLayerData> d = makePtr<ConvLayerData>();
    d->name = lp.name;
    d->type = "Convolution";
    d->params = lp;
    d->inputs = inputs;
    d->outputs = outputs;
    d->initFromParams();
    return d;
}

struct ConvLayerDataRegister
{
    ConvLayerDataRegister()
    {
        registerLayerData("Convolution", createConvLayerData);
    }
};

static ConvLayerDataRegister g_registerConvLayerData;

// Backend-specific layer (ENGINE_NEW): CUDA Convolution wrapper created from ConvLayerData.
// For now it wraps an existing Convolution layer implementation, but is selected through
// the Layer-from-data registry, so we can incrementally move CUDA execution here.
class CUDAConvLayer CV_FINAL : public Layer
{
public:
    CUDAConvLayer(const LayerParams& params, const Ptr<ConvLayerData>& data)
        : Layer(params), convdata(data)
    {
        LayerParams lp = params;
        lp.type = "Convolution";  // CPU fallback until CUDAConvLayer forward is implemented
        cpuFallback = LayerFactory::createLayerInstance(lp.type, lp);
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        std::cout<<"CUDAConvLayer::getMemoryShapes"<<std::endl;
        if (convdata)
            return convdata->getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return cpuFallback ? cpuFallback->getMemoryShapes(inputs, requiredOutputs, outputs, internals)
                        : Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (convdata)
            return convdata->getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals);
        return cpuFallback ? cpuFallback->getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals)
                        : Layer::getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals);
    }

    void finalize(InputArrayOfArrays inputs, OutputArrayOfArrays outputs) CV_OVERRIDE
    {
        if (cpuFallback)
            cpuFallback->netimpl = this->netimpl;
        if (cpuFallback)
            cpuFallback->finalize(inputs, outputs);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals) CV_OVERRIDE
    {
        if (!cpuFallback)
            CV_Error(Error::StsError, "CUDAConvLayer: missing CPU fallback layer implementation");
        cpuFallback->netimpl = this->netimpl;
        cpuFallback->forward(inputs, outputs, internals);
    }

    Ptr<ConvLayerData> convdata;
    Ptr<Layer> cpuFallback;
};

static Ptr<Layer> createCUDAConvLayerFromData(int backendId, const Ptr<LayerOpData>& data)
{
    if (backendId != DNN_BACKEND_CUDA)
        return Ptr<Layer>();
    Ptr<ConvLayerData> convdata = data.dynamicCast<ConvLayerData>();
    if (!convdata)
        return Ptr<Layer>();

    // Simple capability predicate (can be refined later).
    const size_t ksize = convdata->kernel_size.size();
    if (ksize == 0 || ksize > 3)
        return Ptr<Layer>();

    LayerParams lp = convdata->params;
    lp.name = convdata->name;
    lp.type = convdata->type;
    Ptr<Layer> layer = makePtr<CUDAConvLayer>(lp, convdata);
    return layer;
}

struct CUDAConvLayerRegister
{
    CUDAConvLayerRegister()
    {
        // When preferableBackend==DNN_BACKEND_CUDA, Net::finalize() will try this first.
        registerLayerFromData("Convolution", DNN_BACKEND_CUDA, createCUDAConvLayerFromData);
    }
};

static CUDAConvLayerRegister g_registerCudaConvLayer;

// CPU backend layer is implemented as CPUConvLayer (see below). Factory from LayerOpData is defined below.
static Ptr<Layer> createCPUConvLayerFromData(int backendId, const Ptr<LayerOpData>& data);

struct CPUConvLayerRegister
{
    CPUConvLayerRegister()
    {
        registerLayerFromData("Convolution", DNN_BACKEND_OPENCV, createCPUConvLayerFromData);
    }
};

static CPUConvLayerRegister g_registerCpuConvLayer;

}  // namespace

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    bool fusedWeights, fusedBias;
    std::vector<double> weightsMultipliers;
    int groups;
    BaseConvolutionLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations,
                                   padMode, adjust_pads, useWinograd);

        numOutput = -1;
        groups = params.get<int>("group", 1);

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
            pad = Size(pads_begin[1], pads_begin[0]);
            dilation = Size(dilations[1], dilations[0]);

            adjustPad.height = adjust_pads[0];
            adjustPad.width = adjust_pads[1];
        }

        for (int i = 0; i < adjust_pads.size(); i++) {
            CV_Assert(adjust_pads[i] < strides[i]);
        }

        fusedWeights = false;
        fusedBias = false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert((inputs.size() > outputs.size() && blobs.empty()) ||
                  (!inputs.empty() && (blobs.size() == 1 || blobs.size() == 2)));
        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[0];

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims == 3)
        {
            kernel_size.resize(1, kernel_size[0]);
            strides.resize(1, strides[0]);
            dilations.resize(1, dilations[0]);
            pads_begin.resize(1, pads_begin[0]);
            pads_end.resize(1, pads_end[0]);
        }
        CV_Assert(weightShape.dims == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && (input.type() == CV_32F || input.type() == CV_16F));
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[i].type() == input.type());
            CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || inputs[i].dims == 4 || inputs[i].dims == 5) && inputs[i].size[1] == input.size[1]);
            for (int j = 0; j < inputs[i].dims; j++) {
                CV_Assert(inputs[i].size[j] == input.size[j]);
            }
        }

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(inpShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            pad = Size(pads_begin[1], pads_begin[0]);
        }
        fusedWeights = false;
        fusedBias = false;
    }

    bool hasBias() const
    {
        return blobs.size() >= 2;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        if (fusedAdd)   // If the Conv layer has fused Add layer, it cannot fuse other layers.
            return false;

        Ptr<BlankLayer> blank_layer = top.dynamicCast<BlankLayer>();
        if (blank_layer)
            return true;

        Mat w, b;
        top->getScaleShift(w, b);
        if (!w.empty() || !b.empty())
        {
            fuseWeights(w, b);
            fusedWeights = fusedWeights || !w.empty();
            fusedBias = fusedBias || (hasBias() && !w.empty()) || !b.empty();
            return true;
        }
        return false;
    }

    virtual void fuseWeights(const Mat& w_, const Mat& b_) = 0;
};


//TODO: simultaneously convolution and bias addition for cache optimization
class CPUConvLayer CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    enum { VEC_ALIGN = 8, DFT_TYPE = CV_32F };
    Mat weightsMat;  // Used to store weight params. It will be used for layer fusion and memory alignment.
    std::vector<float> biasvec;
    std::vector<float> reluslope;
    Ptr<ActivationLayer> activ;

    Ptr<FastConv> fastConvImpl;
    bool canUseWinograd = false;

#ifdef HAVE_CUDA
    cuda4dnn::ConvolutionConfiguration::FusionMode cudaFusionMode;
    cuda4dnn::ConvolutionConfiguration::ActivationType cudaActType;
    float cuda_relu_slope, cuda_crelu_floor, cuda_crelu_ceil;
    float cuda_power_exp, cuda_power_scale, cuda_power_shift;
#endif

    Ptr<ConvLayerData> convdata;  // ENGINE_NEW: centralized inference data (optional)

    CPUConvLayer(const LayerParams &params) : BaseConvolutionLayerImpl(params)
    {
        // Keep all inference logic in ConvLayerData (used by both classic and ENGINE_NEW paths).
        convdata = makePtr<ConvLayerData>();
        convdata->name = params.name;
        convdata->type = "Convolution";
        convdata->params = params;
        convdata->initFromParams();

#ifdef HAVE_CUDA
        cudaFusionMode = cuda4dnn::ConvolutionConfiguration::FusionMode::NONE;
        cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY;
#endif
    }

    CPUConvLayer(const LayerParams& params, const Ptr<ConvLayerData>& data)
        : BaseConvolutionLayerImpl(params), convdata(data)
    {
#ifdef HAVE_CUDA
        cudaFusionMode = cuda4dnn::ConvolutionConfiguration::FusionMode::NONE;
        cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY;
#endif
    }

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int inpGroupCn = blobs[0].size[1];
        int ksize = inpGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(inpD * inpH * inpW, ksize);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        std::cout<<"CPUConvLayer::getMemoryShapes"<<std::endl;
        CV_Assert(convdata);
        return convdata->getMemoryShapes(inputs, requiredOutputs, outputs, internals);
    }

    void getTypes(const std::vector<MatType> &inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType> &outputs,
                  std::vector<MatType> &internals) const CV_OVERRIDE
    {
        std::cout<<"CPUConvLayer::getTypes"<<std::endl;
        CV_Assert(convdata);
        convdata->getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals);
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerImpl::finalize(inputs_arr, outputs_arr);
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        // prepare weightsMat where each row is aligned and has enough zero padding on the right to
        // use vectorized (i.e. with intrinsics) loops without tail processing
        if (!blobs.empty())
        {
            Mat wm = blobs[0].reshape(1, numOutput);
            if ((wm.step1() % VEC_ALIGN != 0) ||
                !isAligned<VEC_ALIGN * sizeof(float)>(wm.data)
            )
            {
                int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
                Mat wm_buffer = Mat(numOutput, newcols, wm.type());
                Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
                wm_padding.setTo(Scalar::all(0.));
                Mat wm_aligned = wm_buffer.colRange(0, wm.cols);
                wm.copyTo(wm_aligned);
                wm = wm_aligned;
            }
            weightsMat = wm;
        }
        else
        {
            // initialized in .forward()
            weightsMat.release();
        }

        weightsMultipliers.assign(numOutput, 1.0);

        Mat biasMat = hasBias() ? blobs[1].reshape(1, numOutput) : Mat();
        biasvec.resize(numOutput+2);
        if( biasMat.empty() )
        {
            for(int i = 0; i < numOutput; i++ )
                biasvec[i] = 0.f;
        }
        else
        {
            for(int i = 0; i < numOutput; i++ )
                biasvec[i] = biasMat.at<float>(i);
        }
        // Winograd only works when input h and w >= 12.
        canUseWinograd = useWinograd && inputs[0].dims == 4 && inputs[0].size[2] >= 12 && inputs[0].size[3] >= 12;
        if (fastConvImpl && (fastConvImpl->conv_type == CONV_TYPE_WINOGRAD3X3) ^ canUseWinograd)
        {
            fastConvImpl.reset();
        }
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        if ((!activ.empty() && !layer.empty()) || blobs.empty())
            return false;

        activ = layer;
        if (activ.empty())
            reluslope.clear();

        if (activ.empty())
        {
            /* setActivation was called with empty argument => reset all fusions */
#ifdef HAVE_CUDA
            cudaFusionMode = cuda4dnn::ConvolutionConfiguration::FusionMode::NONE;
            cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY;
#endif
        }

        if(IS_DNN_CUDA_TARGET(preferableTarget))
        {
#ifdef HAVE_CUDA
            CV_Assert(cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE ||
                      cudaFusionMode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM);

            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if(!activ_relu.empty())
            {
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::RELU;
                cuda_relu_slope = activ_relu->negativeSlope;
            }

            Ptr<ReLU6Layer> activ_relu6 = activ.dynamicCast<ReLU6Layer>();
            if(!activ_relu6.empty())
            {
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::CLIPPED_RELU;
                cuda_crelu_floor = activ_relu6->minValue;
                cuda_crelu_ceil = activ_relu6->maxValue;
            }

            Ptr<PowerLayer> activ_power = activ.dynamicCast<PowerLayer>();
            if (!activ_power.empty())
            {
                cuda_power_scale = activ_power->scale;
                cuda_power_shift = activ_power->shift;
                cuda_power_exp = activ_power->power;
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::POWER;
            }

            Ptr<TanHLayer> activ_tanh = activ.dynamicCast<TanHLayer>();
            if(!activ_tanh.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::TANH;

            Ptr<SigmoidLayer> activ_sigmoid = activ.dynamicCast<SigmoidLayer>();
            if(!activ_sigmoid.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::SIGMOID;

            Ptr<SwishLayer> activ_swish = activ.dynamicCast<SwishLayer>();
            if(!activ_swish.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::SWISH;

            Ptr<MishLayer> activ_mish = activ.dynamicCast<MishLayer>();
            if(!activ_mish.empty())
                cudaActType = cuda4dnn::ConvolutionConfiguration::ActivationType::MISH;

            if (cudaActType == cuda4dnn::ConvolutionConfiguration::ActivationType::IDENTITY)
            {
                /* no activation fused */
                activ.reset();
            }
            else
            {
                /* activation was fused */
                if (cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE) /* no previous fusion */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ACTIVATION; /* now activation */
                else if (cudaFusionMode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM) /* previously eltwise was fused */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION; /* now activation on eltwise output */
            }
#endif  // HAVE_CUDA
        }
        fusedActivation = !activ.empty();
        return fusedActivation;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        if (fusedAdd)   // If the Conv layer has fused Add layer, it cannot fuse other layers.
            return false;

        if(IS_DNN_CUDA_TARGET(preferableTarget))
        {
#ifdef HAVE_CUDA
            Ptr<EltwiseLayer> eltwise = top.dynamicCast<EltwiseLayer>();
            Ptr<NaryEltwiseLayer> naryEltwise = top.dynamicCast<NaryEltwiseLayer>();
            if (!eltwise.empty() || !naryEltwise.empty())
            {
                /* we also need to check that the eltwise input does not require shortcut mechanism
                 * it's difficult to verify it here but we hope that `fuseLayers` has done the check already
                 */
                if (cudaFusionMode == ConvolutionConfiguration::FusionMode::NONE)
                {
                    /* no previous fusion */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ELTWISE_SUM; /* now eltwise */
                    return true;
                }
                else if(cudaFusionMode == ConvolutionConfiguration::FusionMode::ACTIVATION)
                {
                    /* previously an activation was fused */
                    cudaFusionMode = ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM;
                    return true;
                }
                return false;
            }
#endif  // HAVE_CUDA
        }
        return BaseConvolutionLayerImpl::tryFuse(top);
    }

    void fuseWeights(const Mat& w_, const Mat& b_) CV_OVERRIDE
    {
        // Convolution weights have OIHW data layout. Parameters fusion in case of
        // (conv(I) + b1 ) * w + b2
        // means to replace convolution's weights to [w*conv(I)] and bias to [b1 * w + b2]
        const int outCn = weightsMat.size[0];
        Mat w = w_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, outCn, CV_32F, Scalar(b_.at<float>(0))) : b_;
        CV_Assert_N(!weightsMat.empty(), biasvec.size() == outCn + 2,
                    w.empty() || outCn == w.total(), b.empty() || outCn == b.total());

        if (!w.empty())
        {
            // Keep origin weights unchanged.
            if (weightsMat.data == blobs[0].data)
                weightsMat = weightsMat.clone();

            Mat originWeights = blobs[0].reshape(1, outCn);
            for (int i = 0; i < outCn; ++i)
            {
                double wi = w.at<float>(i);
                weightsMultipliers[i] *= wi;
                cv::multiply(originWeights.row(i), weightsMultipliers[i], weightsMat.row(i));
                biasvec[i] *= wi;
            }
        }

        if (!b.empty())
        {
            for (int i = 0; i < outCn; ++i)
                biasvec[i] += b.at<float>(i);
        }
        biasvec[outCn] = biasvec[outCn+1] = biasvec[outCn-1];
    }

    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> >&, std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &,
                                      const std::vector<Ptr<BackendWrapper> > &,
                                      const std::vector<Ptr<BackendNode> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &,
                                        const std::vector<Ptr<BackendNode> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >&,
                                       const std::vector<Ptr<BackendNode> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        int outCn = blobs.empty() ? inputs[1].size[0] : blobs[0].size[0];
        // Need to align non-const blobs
        bool variableWeight = false;
        if (blobs.empty())
        {
            variableWeight = true;
            Mat wm = inputs[1].reshape(1, outCn);
            if (wm.data != weightsMat.data)
            {
                int newcols = (int)alignSize(wm.step1(), VEC_ALIGN);
                Mat wm_buffer = Mat(numOutput, newcols, wm.type());
                Mat wm_padding = wm_buffer.colRange(wm.cols, newcols);
                wm_padding.setTo(Scalar::all(0.));
                weightsMat = wm_buffer.colRange(0, wm.cols);

                wm.copyTo((const Mat&)weightsMat);
                if (inputs.size() > 2)
                {
                    Mat biasMat = inputs[2].reshape(1, outCn);
                    biasMat.col(0).copyTo(biasvec);
                }
                biasvec.resize(outCn + 2, 0);
            }
        }
        /*if (inputs[0].dims > 3) {
            printf("conv %s: input (%d x %d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2], inputs[0].size[3],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }
        else {
            printf("conv %s: input (%d x %d x %d), kernel (%d x %d), pad (%d x %d), stride (%d x %d), dilation (%d x %d)\n",
                   name.c_str(), inputs[0].size[0], inputs[0].size[1], inputs[0].size[2],
                   kernel.width, kernel.height, pad.width, pad.height,
                   stride.width, stride.height, dilation.width, dilation.height);
        }*/
        int inpGroupCn = blobs.empty() ? inputs[1].size[1] : blobs[0].size[1];
        CV_Assert_N(inputs.size() >= (size_t)1, inputs[0].size[1] % inpGroupCn == 0,
                    outputs.size() == 1, inputs[0].data != outputs[0].data);

        int ngroups = inputs[0].size[1] / inpGroupCn;
        CV_Assert(outputs[0].size[1] % ngroups == 0);

        reluslope.clear();
        if( activ )
        {
            Ptr<ReLULayer> activ_relu = activ.dynamicCast<ReLULayer>();
            if( !activ_relu.empty() )
            {
                reluslope.assign(outCn+2, activ_relu->negativeSlope);
            }

            Ptr<ChannelsPReLULayer> activ_chprelu = activ.dynamicCast<ChannelsPReLULayer>();
            if( !activ_chprelu.empty() )
            {
                const Mat& m = activ_chprelu->blobs[0];
                CV_Assert(m.isContinuous() && m.type() == CV_32F && (int)m.total() == outCn);
                const float* mdata = m.ptr<float>();
                reluslope.resize(outCn+2);
                std::copy(mdata, mdata + outCn, reluslope.begin());
                reluslope[outCn] = reluslope[outCn+1] = reluslope[outCn-1];
            }
        }

        {
            int nstripes = std::max(getNumThreads(), 1);
            int conv_dim = CONV_2D;
            if (inputs[0].dims == 3)
                conv_dim = CONV_1D;
            if (inputs[0].dims == 5)
                conv_dim = CONV_3D;

            // Initialization of FastCovn2d, pack weight.
            if (!fastConvImpl || variableWeight)
            {
                int K = outputs[0].size[1];
                int C = inputs[0].size[1];

                CV_Assert(outputs[0].size[1] % ngroups == 0);
                fastConvImpl = initFastConv(weightsMat, &biasvec[0], ngroups, K, C, kernel_size, strides,
                                            dilations, pads_begin, pads_end, conv_dim,
                                            preferableTarget == DNN_TARGET_CPU_FP16, canUseWinograd);
                // This is legal to release weightsMat here as this is not used anymore for
                // OpenCV inference. If network needs to be reinitialized (new shape, new backend)
                // a new version of weightsMat is created at .finalize() from original weights
                weightsMat.release();
            }

            runFastConv(inputs[0], outputs[0], fastConvImpl, nstripes, activ, reluslope, fusedAdd);
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        // TODO: extract bias from inputs and pass it
        CV_Assert(inputs.size() == 1 || inputs.size() == 2);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        CV_Assert(!blobs.empty());
        const auto output_feature_maps = blobs[0].size[0];
        const auto input_feature_maps = input_shape[1];
        const auto input_feature_maps_per_group = blobs[0].size[1];
        const auto groups = input_feature_maps / input_feature_maps_per_group;

        ConvolutionConfiguration config;

        if (input_shape.size() == 3)
        {
            // Conv1D
            // We add an extra dim for input and output tensors, because CuDNN doesn't support convolution with 3D tensors
            input_shape.insert(std::end(input_shape) - 1, 1);
            output_shape.insert(std::end(output_shape) - 1, 1);

            // Do the similar thing for the other parameters
            pads_begin.insert(std::begin(pads_begin), 0);
            pads_end.insert(std::begin(pads_end), 0);
            strides.insert(std::begin(strides), 1);
            dilations.insert(std::begin(dilations), 1);
            kernel_size.insert(std::begin(kernel_size), 1);
        }
        config.kernel_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.dilations.assign(std::begin(dilations), std::end(dilations));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = ConvolutionConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by ConvolutionLayer");
        }

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.output_shape.assign(std::begin(output_shape), std::end(output_shape));
        config.groups = groups;

        config.fusion_mode = cudaFusionMode;
        config.activation_type = cudaActType;
        config.relu_negative_slope = cuda_relu_slope;
        config.crelu_floor = cuda_crelu_floor;
        config.crelu_ceil = cuda_crelu_ceil;
        config.power_exp = cuda_power_exp;
        config.power_scale = cuda_power_scale;
        config.power_shift = cuda_power_shift;

        Mat filtersMat = fusedWeights ? weightsMat : blobs[0];
        Mat biasMat = (hasBias() || fusedBias) ? Mat(output_feature_maps, 1, CV_32F, biasvec.data()) : Mat();
        if (countNonZero(biasMat) == 0)
            biasMat = Mat();

        return make_cuda_node<cuda4dnn::ConvolutionOp>(
            preferableTarget, std::move(context->stream), std::move(context->cudnn_handle), config, filtersMat, biasMat);
    }
#endif  // HAVE_CUDA

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size() || inputs.size() == outputs.size() + blobs.size());

        int64 flops = 0;
        int karea = std::accumulate(kernel_size.begin(), kernel_size.end(), 1, std::multiplies<size_t>());
        for (int i = 0; i < outputs.size(); i++)
        {
            flops += total(outputs[i])*(CV_BIG_INT(2)*karea*inputs[i][1] + 1);
        }

        return flops;
    }
};

namespace {
static Ptr<Layer> createCPUConvLayerFromData(int backendId, const Ptr<LayerOpData>& data)
{
    if (backendId != DNN_BACKEND_OPENCV)
        return Ptr<Layer>();
    Ptr<ConvLayerData> convdata = data.dynamicCast<ConvLayerData>();
    if (!convdata)
        return Ptr<Layer>();
    LayerParams lp = convdata->params;
    lp.name = convdata->name;
    lp.type = convdata->type;
    return makePtr<CPUConvLayer>(lp, convdata);
}
}  // namespace

class DeConvolutionLayerImpl CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    Mat weightsMat, biasesMat;
    UMat umat_weights;
    UMat umat_biases;

    DeConvolutionLayerImpl(const LayerParams& params) : BaseConvolutionLayerImpl(params) {}

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int outCn = outShape[1];
        int outGroupCn = outCn / groups;
        int ksize = outGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(ksize, inpD * inpH * inpW);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
        {
            /* only deconvolution 2d and 3d supported */
            if (kernel_size.size() == 2 || kernel_size.size() == 3)
                return true;

            return false;
        }
        return kernel_size.size() == 2 && backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() != 0);

        int outCn = numOutput;
        if (outCn < 0) {
            CV_Assert(inputs.size() > 1 || !blobs.empty());
            MatShape weightShape = blobs.empty() ? inputs[1] : blobs[0].shape();
            outCn = weightShape[1]*groups;
        }
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);  // batch
        outShape.push_back(outCn);
        if (padMode.empty())
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + kernel_size[i] - pads_begin[i] - pads_end[i] + adjust_pads[i]);
        }
        else if (padMode == "VALID")
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + kernel_size[i] + adjust_pads[i]);
        }
        else if (padMode == "SAME")
        {
            for (int i = 0; i < kernel_size.size(); i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + 1 + adjust_pads[i]);
        }
        else
            CV_Error(Error::StsError, "Unsupported padding mode " + padMode);

        CV_Assert(outCn % blobs[0].size[1] == 0);

        int inpCn = inputs[0][1];
        CV_Assert(inpCn % groups == 0 && outCn % groups == 0);
        CV_Assert(blobs[0].size[0] == inpCn);

        outputs.resize(1, MatShape(outShape));

        if (!is1x1())
            internals.push_back(computeColRowShape(inputs[0], outputs[0]));

        return false;
    }

    void getTypes(const std::vector<MatType> &inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType> &outputs,
                  std::vector<MatType> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, CV_32F);
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerImpl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() > 1 || !blobs.empty());

        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[1]*groups;

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(outShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in deconvolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }

        weightsMultipliers.assign(numOutput, 1.0);

        if (weightsMat.empty() && !blobs.empty()) {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
        }

        if (biasesMat.empty() && blobs.size() >= 2) {
            biasesMat = blobs[1].reshape(1, numOutput);
        }
    }

    void fuseWeights(const Mat& w_, const Mat& b_) CV_OVERRIDE
    {
        Mat w = w_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(b_.at<float>(0))) : b_;

        CV_Assert_N(!weightsMat.empty(),
                     w.empty() || numOutput == w.total(),
                     b.empty() || numOutput == b.total());

        if (!w.empty())
        {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
            weightsMat = weightsMat.reshape(1, numOutput);
            for (int i = 0; i < numOutput; ++i)
            {
                double wi = w.at<float>(i);
                weightsMultipliers[i] *= wi;
                cv::multiply(weightsMat.row(i), weightsMultipliers[i], weightsMat.row(i));
                biasesMat.at<float>(i) *= wi;
            }
            weightsMat = weightsMat.reshape(1, weightsMat.total() / blobs[0].size[0]);
        }

        if (!b.empty())
        {
            cv::add(biasesMat, b.reshape(1, numOutput), biasesMat);
        }
    }

    class MatMulInvoker : public ParallelLoopBody
    {
    public:
        MatMulInvoker(const Mat& a, const Mat& b, Mat& c, int nstripes)
        {
            a_ = &a;
            b_ = &b;
            c_ = &c;
            nstripes_ = nstripes;
            useAVX = checkHardwareSupport(CPU_AVX);
            useAVX2 = checkHardwareSupport(CPU_AVX2);
            useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;
            useRVV = checkHardwareSupport(CPU_RVV);
            useLASX = checkHardwareSupport(CPU_LASX);
        }

        void operator()(const Range& range_) const CV_OVERRIDE
        {
            int stripeSize = (int)alignSize((b_->cols + nstripes_ - 1)/nstripes_, 16);
            Range range(range_.start*stripeSize, std::min(range_.end*stripeSize, b_->cols));
            int mmax = a_->rows;
            int nmax = range.end - range.start;
            int kmax = a_->cols;
            int m, n, k;
            const float* aptr = a_->ptr<float>();
            const float* bptr = b_->ptr<float>() + range.start;
            float* cptr = c_->ptr<float>() + range.start;
            size_t astep = a_->step1();
            size_t bstep = b_->step1();
            size_t cstep = c_->step1();

        #if CV_TRY_AVX512_SKX
            if( useAVX512 )
                opt_AVX512_SKX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX2
            if( useAVX2 )
                opt_AVX2::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX
            if( useAVX )
                opt_AVX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_RVV && CV_RVV
            if( useRVV ) {
                opt_RVV::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            }
            else
        #endif
        #if CV_TRY_LASX
            if( useLASX )
                opt_LASX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
            for( m = 0; m < mmax; m += 2 )
            {
                float* dst0 = cptr + cstep*m;
                float* dst1 = cptr + cstep*std::min(m+1, mmax-1);
                const float* aptr0 = aptr + astep*m;
                const float* aptr1 = aptr + astep*std::min(m+1, mmax-1);

                for( n = 0; n < nmax; n++ )
                {
                    dst0[n] = 0.f;
                    dst1[n] = 0.f;
                }

                for( k = 0; k < kmax; k += 4 )
                {
                    float alpha00 = aptr0[k];
                    float alpha01 = aptr1[k];
                    float alpha10 = 0.f, alpha11 = 0.f;
                    float alpha20 = 0.f, alpha21 = 0.f;
                    float alpha30 = 0.f, alpha31 = 0.f;
                    const float* bptr0 = bptr + k*bstep;
                    const float* bptr1 = bptr0;
                    const float* bptr2 = bptr0;
                    const float* bptr3 = bptr0;

                    if( k+1 < kmax )
                    {
                        alpha10 = aptr0[k+1];
                        alpha11 = aptr1[k+1];
                        bptr1 = bptr0 + bstep;
                        if( k+2 < kmax )
                        {
                            alpha20 = aptr0[k+2];
                            alpha21 = aptr1[k+2];
                            bptr2 = bptr1 + bstep;
                            if( k+3 < kmax )
                            {
                                alpha30 = aptr0[k+3];
                                alpha31 = aptr1[k+3];
                                bptr3 = bptr2 + bstep;
                            }
                        }
                    }
                    n = 0;

                #if CV_SIMD128
                    v_float32x4 a00 = v_setall_f32(alpha00);
                    v_float32x4 a01 = v_setall_f32(alpha01);
                    v_float32x4 a10 = v_setall_f32(alpha10);
                    v_float32x4 a11 = v_setall_f32(alpha11);
                    v_float32x4 a20 = v_setall_f32(alpha20);
                    v_float32x4 a21 = v_setall_f32(alpha21);
                    v_float32x4 a30 = v_setall_f32(alpha30);
                    v_float32x4 a31 = v_setall_f32(alpha31);

                    for( ; n <= nmax - 4; n += 4 )
                    {
                        v_float32x4 d0 = v_load(dst0 + n);
                        v_float32x4 d1 = v_load(dst1 + n);
                        v_float32x4 b0 = v_load(bptr0 + n);
                        v_float32x4 b1 = v_load(bptr1 + n);
                        v_float32x4 b2 = v_load(bptr2 + n);
                        v_float32x4 b3 = v_load(bptr3 + n);
                        // TODO try to improve pipeline width
                        d0 = v_fma(b0, a00, d0);
                        d1 = v_fma(b0, a01, d1);
                        d0 = v_fma(b1, a10, d0);
                        d1 = v_fma(b1, a11, d1);
                        d0 = v_fma(b2, a20, d0);
                        d1 = v_fma(b2, a21, d1);
                        d0 = v_fma(b3, a30, d0);
                        d1 = v_fma(b3, a31, d1);
                        v_store(dst0 + n, d0);
                        v_store(dst1 + n, d1);
                    }
                #endif

                    for( ; n < nmax; n++ )
                    {
                        float b0 = bptr0[n];
                        float b1 = bptr1[n];
                        float b2 = bptr2[n];
                        float b3 = bptr3[n];
                        float d0 = dst0[n] + alpha00*b0 + alpha10*b1 + alpha20*b2 + alpha30*b3;
                        float d1 = dst1[n] + alpha01*b0 + alpha11*b1 + alpha21*b2 + alpha31*b3;
                        dst0[n] = d0;
                        dst1[n] = d1;
                    }
                }
            }
        }

        const Mat *a_, *b_;
        Mat* c_;
        int nstripes_;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
        bool useRVV;
        bool useLASX;
    };

    class Col2ImInvoker : public cv::ParallelLoopBody
    {
    public:
        const float* data_col;
        const float* biasvec;
        int channels, height, width;
        int kernel_h, kernel_w;
        int pad_h, pad_w;
        int stride_h, stride_w;
        float* data_im;
        int height_col, width_col;
        int nstripes;
        bool is1x1;

        Col2ImInvoker()
            : data_col(0), biasvec(0), channels(0), height(0), width(0),
              kernel_h(0), kernel_w(0), pad_h(0), pad_w(0), stride_h(0), stride_w(0), data_im(0),
              height_col(0), width_col(0), nstripes(0), is1x1(0)
        {}

        static void run(const float* data_col,
                        int channels, int height, int width,
                        int kernel_h, int kernel_w,
                        int pad_h, int pad_w,
                        int stride_h, int stride_w,
                        int height_col, int width_col,
                        float* data_im,
                        const float* biasvec,
                        bool is1x1)
        {
            const int nstripes = getNumThreads();

            Col2ImInvoker t;
            t.data_col = data_col;
            t.data_im = data_im;
            t.channels = channels; t.height = height; t.width = width;
            t.kernel_h = kernel_h; t.kernel_w = kernel_w;
            t.pad_h = pad_h; t.pad_w = pad_w;
            t.stride_h = stride_h; t.stride_w = stride_w;
            t.height_col = height_col;
            t.width_col = width_col;
            t.nstripes = nstripes;
            t.is1x1 = is1x1;
            t.biasvec = biasvec;

            parallel_for_(Range(0, nstripes), t, nstripes);
        }

        virtual void operator ()(const Range &r) const CV_OVERRIDE
        {
            const float* data_col_ = data_col;
            float* data_im_ = data_im;
            int coeff_h = (1 - stride_h * kernel_w * height_col) * width_col;
            int coeff_w = (1 - stride_w * height_col * width_col);
            size_t total = (size_t)channels * height * width;
            size_t stripeSize = (total + nstripes - 1)/nstripes;
            size_t startIndex = r.start*stripeSize;
            size_t endIndex = std::min(r.end*stripeSize, total);
            int w = (int)(startIndex % width + pad_w);
            int h = (int)((startIndex / width) % height + pad_h);
            int c = (int)(startIndex / (width * height));
            int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            int h_col_end = std::min(h / stride_h + 1, height_col);
            int plane_size_col = height_col * width_col;
            int offset = (c * kernel_h * kernel_w + h * kernel_w + w) * plane_size_col;
            bool is1x1_ = is1x1;
            const float* biasvec_ = biasvec;

            for (size_t index = startIndex; index < endIndex; index++)
            {
                // compute the start and end of the output
                int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
                int w_col_end = std::min(w / stride_w + 1, width_col);
                float val;

                if( is1x1_ )
                    val = data_im_[index];
                else
                {
                    val = 0.f;
                    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
                        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                            val += data_col_[offset + h_col * coeff_h + w_col * coeff_w];
                        }
                    }
                }
                data_im_[index] = val + biasvec_[c];

                offset += plane_size_col;
                if( ++w >= width + pad_w )
                {
                    w = (int)((index + 1)% width + pad_w);
                    h = (int)(((index + 1) / width) % height + pad_h);
                    c = (int)((index + 1) / (width * height));
                    h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
                    h_col_end = std::min(h / stride_h + 1, height_col);
                    offset = (c * kernel_h * kernel_w + h * kernel_w + w) * plane_size_col;
                }
            }
        }
    };

    bool forward_ocl(InputArrayOfArrays, OutputArrayOfArrays, OutputArrayOfArrays)
    {
        // OpenCL path removed (CPU/CUDA-only build).
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // For some reason, tests for deconvolution fail;
        // Also, the current implementation is super-inefficient,
        // Just disabled it. Need to rewrite it and then uncomment back these lines
        //CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
        //           forward_ocl(inputs_arr, outputs_arr, internals_arr));

        if (inputs_arr.depth(0) == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        auto kind = outputs_arr.kind();
        std::vector<Mat> inputs, internals;
        inputs_arr.getMatVector(inputs);
        internals_arr.getMatVector(internals);

        int outCn = numOutput;
        int inpCn = inputs[0].size[1];
        bool is1x1flag = is1x1();
        int nstripes = getNumThreads();
        /*CV_Assert(outputs.size() == 1);
        CV_Assert(inputs[0].size[0] == outputs[0].size[0]);
        CV_Assert(outCn == outputs[0].size[1]);*/

        if (weightsMat.empty() || inputs.size() >= 2) {
            Mat inpWeights = !blobs.empty() ? blobs[0] : inputs[1];
            transpose(inpWeights.reshape(1, inpCn), weightsMat);
        }

        if (biasesMat.empty() || inputs.size() >= 3) {
            Mat inpBias = blobs.size() >= 2 ? blobs[1] : inputs.size() >= 3 ? inputs[2] : Mat();
            Mat biasesMat_ = !inpBias.empty() ? inpBias.reshape(1, outCn) : Mat::zeros(outCn, 1, CV_32F);
            biasesMat_.copyTo(biasesMat);
        }

        /*printf("DeConvolution Input: ");
        pprint(std::cout, inputs[0], 0, 3, 100, '[');
        printf("\nDeConvolution Weights: ");
        pprint(std::cout, weightsMat, 0, 3, 100, '[');
        printf("\nDeConvolution Bias: ");
        pprint(std::cout, biasesMat, 0, 3, 100, '[');
        printf("\n");*/

        //for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ii = 0;
            int inpGroupCn = inpCn / groups;
            int outGroupCn = outCn / groups;
            const Mat& inp = inputs[ii];
            MatShape outshape = outputs_arr.shape(0);
            CV_Assert(outshape.dims == inp.dims);
            CV_Assert(outshape[0] == inp.size[0]);
            CV_Assert(outshape[1] == outCn);
            Mat out;
            if (kind == _InputArray::STD_VECTOR_MAT) {
                out = outputs_arr.getMat(0);
            }
            else {
                out.create(outshape, inp.type());
            }
            int numImg = inp.size[0];
            int inpH = inp.size[2], inpW = inp.size[3];
            int outH = out.size[2], outW = out.size[3];

            Mat convBlob = inputs[ii].reshape(1, numImg*inpCn);
            Mat decnBlob = out.reshape(1, numImg*outCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < groups; g++)
                {
                    Mat dstMat = decnBlob.rowRange(_Range((g + n * groups) * outGroupCn, outGroupCn));
                    Mat &colMat = is1x1flag ? dstMat : internals[0];

                    Mat convMat = convBlob.rowRange(_Range((g + n * groups) * inpGroupCn, inpGroupCn));
                    Mat wghtMat = weightsMat.colRange(_Range(g * inpGroupCn, inpGroupCn));
                    Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));

                    //gemm(wghtMat, convMat, 1, colMat, 0, colMat, 0);
                    MatMulInvoker mminvoker(wghtMat, convMat, colMat, nstripes);
                    parallel_for_(Range(0, nstripes), mminvoker, nstripes);

                    Col2ImInvoker::run(colMat.ptr<float>(), outGroupCn, outH, outW,
                                       kernel.height, kernel.width, pad.height, pad.width,
                                       stride.height, stride.width, inpH, inpW, dstMat.ptr<float>(),
                                       curBiasMat.ptr<float>(), is1x1flag);
                }
            }
            if (kind == _InputArray::STD_VECTOR_UMAT) {
                std::vector<UMat>& u_outputs = outputs_arr.getUMatVecRef();
                out.copyTo(u_outputs[0]);
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        CV_Assert(!blobs.empty());
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(inputs.size() == 1);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        const auto output_feature_maps = numOutput;
        const auto output_feature_maps_per_group = blobs[0].size[1];
        const auto groups = output_feature_maps / output_feature_maps_per_group;

        TransposeConvolutionConfiguration config;
        config.kernel_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.dilations.assign(std::begin(dilations), std::end(dilations));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by DeconvolutionLayer");
        }

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.output_shape.assign(std::begin(output_shape), std::end(output_shape));
        config.groups = groups;

        CV_Assert(blobs.size() >= 1);
        Mat filtersMat = fusedWeights ? weightsMat.t() : blobs[0];

        Mat biasMat = (hasBias() || fusedBias) ? biasesMat : Mat();
        if (countNonZero(biasMat) == 0)
            biasMat = Mat();

        return make_cuda_node<cuda4dnn::TransposeConvolutionOp>(
            preferableTarget, std::move(context->stream), std::move(context->cudnn_handle), config, filtersMat, biasMat);
    }
#endif  // HAVE_CUDA

    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &,
                                      const std::vector<Ptr<BackendWrapper> > &,
                                      const std::vector<Ptr<BackendNode> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &,
                                        const std::vector<Ptr<BackendNode> >&) CV_OVERRIDE
    {
        return Ptr<BackendNode>();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size());

        float flops = 0;
        int outChannels = blobs[0].size[0];
        size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                       1, std::multiplies<size_t>());

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += CV_BIG_INT(2)*outChannels*karea*total(inputs[i]);
        }

        return flops;
    }
};

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<CPUConvLayer> l(new CPUConvLayer(params));
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    return Ptr<BaseConvolutionLayer>(new DeConvolutionLayerImpl(params));
}

}
}
