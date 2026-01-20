// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
#ifndef OPENCV_DNN_SRC_LAYER_DATA_WRAPPER_HPP
#define OPENCV_DNN_SRC_LAYER_DATA_WRAPPER_HPP

#include "precomp.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
namespace detail {

// Generic fallback LayerData implementation which delegates shape/type inference to a temporary
// executable Layer instance created from the legacy LayerFactory registry.
class FallbackLayerHelper CV_FINAL : public cv::dnn::LayerHelper
{
public:
    explicit FallbackLayerHelper(const LayerParams& lp) : params_(lp) {}

    String type() const CV_OVERRIDE { return params_.type; }
    String name() const CV_OVERRIDE { return params_.name; }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        LayerParams lp = params_;
        Ptr<Layer> layer = LayerFactory::createLayerInstance(lp.type, lp);
        if (!layer)
            CV_Error(Error::StsError, "Can't create layer '" + lp.name + "' of type '" + lp.type + "' for fallback shape inference");
        return layer->getMemoryShapes(inputs, requiredOutputs, outputs, internals);
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        LayerParams lp = params_;
        Ptr<Layer> layer = LayerFactory::createLayerInstance(lp.type, lp);
        if (!layer)
            CV_Error(Error::StsError, "Can't create layer '" + lp.name + "' of type '" + lp.type + "' for fallback type inference");
        layer->getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals);
    }

    LayerParams getParams() const CV_OVERRIDE { return params_; }

private:
    LayerParams params_;
};

// A non-executable Layer wrapper around LayerData.
// Used in the "abstract graph" during import: supports shape/type inference but has no forward().
class DataOnlyLayer CV_FINAL : public cv::dnn::Layer
{
public:
    explicit DataOnlyLayer(const Ptr<cv::dnn::LayerHelper>& helper) : helper_(helper)
    {
        CV_Assert(helper_);
        name = helper_->name();
        type = helper_->type();
    }

    std::vector<Ptr<Graph> >* subgraphs() const CV_OVERRIDE
    {
        // Some ONNX ops (e.g. If) store subgraphs in the layer instance.
        // DataOnlyLayer must support it during import before executable layers are instantiated.
        return &subgraphs_;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        return helper_->getMemoryShapes(inputs, requiredOutputs, outputs, internals);
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        helper_->getTypes(inputs, requiredOutputs, requiredInternals, outputs, internals);
    }

    void forward(InputArrayOfArrays, OutputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented,
                 "DNN: DataOnlyLayer cannot execute forward(). Call Net::finalize() to instantiate executable layers.");
    }

    Ptr<cv::dnn::LayerHelper> getLayerHelper() const { return helper_; }

private:
    Ptr<cv::dnn::LayerHelper> helper_;
    mutable std::vector<Ptr<Graph> > subgraphs_;
};

}  // namespace detail
CV__DNN_INLINE_NS_END
}  // namespace dnn
}  // namespace cv

#endif  // OPENCV_DNN_SRC_LAYER_DATA_WRAPPER_HPP
