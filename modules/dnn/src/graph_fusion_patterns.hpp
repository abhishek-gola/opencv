// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_GRAPH_FUSION_PATTERNS_HPP
#define OPENCV_DNN_GRAPH_FUSION_PATTERNS_HPP

#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Outcome of a successful fusion; untouched when the helper returns false.
struct FuseResult
{
    int layer_idx = -1;
    std::vector<Arg> new_inputs;
    std::vector<Arg> removed_args;
};

// Per-graph state threaded through every fusion helper. Helpers that add new
// graph args (merged weights/biases) mutate `usecounts` and `producer_of`.
struct FusionContext
{
    Net::Impl* netimpl;
    std::vector<Ptr<Layer> >& newprog;
    std::vector<int>& producer_of;
    std::vector<int>& usecounts;
};

// Transpose(a) -> Transpose(b) => single Transpose(a∘b), or drop both if identity.
bool tryFuseTransposeTranspose(const FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r);

// Conv -> BatchNorm => Conv. Requires BN's scale/bias/mean/var to have been
// frozen by constArgs() (so BN->inputs.size() == 1).
bool tryFuseConvBatchNorm(const FusionContext& ctx, const Ptr<Layer>& layer,
                          const std::vector<Arg>& inputs, FuseResult& r);

// Conv + residual-Add/Sum => Conv; merges into the later-scheduled operand.
bool tryFuseConvAddResidual(const FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r);

// Split -> N identical Convs -> Concat => single grouped Conv (ngroups=N).
bool tryFuseSplitConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r);

// N Convs sharing the same input -> Concat => single wider Conv.
// Per-branch kernels may differ; smaller kernels are zero-padded to the max.
bool tryFuseParallelConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r);

// Conv -> Activation => Conv with activation fused into its epilogue.
bool tryFuseConvActivation(const FusionContext& ctx, const Ptr<Layer>& layer,
                           const std::vector<Arg>& inputs, FuseResult& r);

CV__DNN_INLINE_NS_END
}}

#endif
