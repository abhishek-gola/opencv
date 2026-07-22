// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// ONNX Scan operator. The scan body subgraph is iterated over the scan axis of the
// scan inputs; the interpreter (Net::Impl::forwardGraph) drives the loop and stacks
// the per-iteration scan outputs. This layer only holds the body and its attributes.
class ScanLayerImpl CV_FINAL : public ScanLayer
{
public:
    explicit ScanLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        num_scan_inputs = params.get<int>("num_scan_inputs");
        readIntList(params, "scan_input_axes", input_axes);
        readIntList(params, "scan_output_axes", output_axes);
        readIntList(params, "scan_input_directions", input_dirs);
        readIntList(params, "scan_output_directions", output_dirs);
    }

    std::vector<Ptr<Graph> >* subgraphs() const CV_OVERRIDE { return &body_; }
    bool dynamicOutputShapes() const CV_OVERRIDE { return true; }

    bool getMemoryShapes(const std::vector<MatShape>&,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        outputs.assign(std::max(1, requiredOutputs), MatShape());
        internals.clear();
        return false;
    }

    int numScanInputs() const CV_OVERRIDE { return num_scan_inputs; }
    const std::vector<int>& scanInputAxes() const CV_OVERRIDE { return input_axes; }
    const std::vector<int>& scanOutputAxes() const CV_OVERRIDE { return output_axes; }
    const std::vector<int>& scanInputDirections() const CV_OVERRIDE { return input_dirs; }
    const std::vector<int>& scanOutputDirections() const CV_OVERRIDE { return output_dirs; }

private:
    static void readIntList(const LayerParams& params, const std::string& name, std::vector<int>& out)
    {
        out.clear();
        if (!params.has(name)) return;
        const DictValue& v = params.get(name);
        out.resize(v.size());
        for (int i = 0; i < v.size(); ++i) out[i] = v.get<int>(i);
    }

    int num_scan_inputs = 0;
    std::vector<int> input_axes, output_axes, input_dirs, output_dirs;
    mutable std::vector<Ptr<Graph> > body_;
};

Ptr<ScanLayer> ScanLayer::create(const LayerParams& params)
{
    return makePtr<ScanLayerImpl>(params);
}
CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
