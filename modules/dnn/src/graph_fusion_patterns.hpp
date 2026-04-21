// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_GRAPH_FUSION_PATTERNS_HPP
#define OPENCV_DNN_GRAPH_FUSION_PATTERNS_HPP

#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Result slot populated by a successful fusion helper.
struct FuseResult
{
    int layer_idx = -1;             // index of the surviving layer in `newprog`
    std::vector<Arg> new_inputs;    // if non-empty, replaces the surviving layer's inputs
    std::vector<Arg> removed_args;  // args whose use-count must be cleared
};

// Bundles the per-graph state threaded through every fusion helper.
// `usecounts` and `producer_of` may be mutated by helpers that create new
// graph args (e.g. merged weight/bias tensors for multi-branch conv fusions).
struct FusionContext
{
    Net::Impl* netimpl;
    std::vector<Ptr<Layer> >& newprog;
    std::vector<int>& producer_of;
    std::vector<int>& usecounts;
};

// Each helper returns true iff its pattern matched the current layer. In that
// case `r` is populated with the fusion outcome; otherwise `r` is untouched.

// Transpose(perm1) -> Transpose(perm2)  =>  Transpose(perm1 ∘ perm2)
// (or drops both if the composed permutation is the identity).
bool tryFuseTransposeTranspose(const FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r);

// Fold BatchNorm into the preceding Conv. Requires BN's scale/bias/mean/var
// to already be frozen by constArgs() so that BN->inputs has shrunk to size 1.
bool tryFuseConvBatchNorm(const FusionContext& ctx, const Ptr<Layer>& layer,
                          const std::vector<Arg>& inputs, FuseResult& r);

// Fold a residual Add/Sum into the Conv that produces one of its operands
// (the later-scheduled operand, so the other is already materialized).
bool tryFuseConvAddResidual(const FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r);

// Split -> {N identical Convs} -> Concat  =>  a single grouped Conv (ngroups=N).
bool tryFuseSplitConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r);

// {N Convs sharing the same input} -> Concat  =>  a single wider Conv.
// Matches SqueezeNet-style "fire" modules; supports per-branch kernel sizes
// by zero-padding smaller kernels to the max so centered weights are preserved.
bool tryFuseParallelConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r);

// Fuse an activation into the preceding Conv.
bool tryFuseConvActivation(const FusionContext& ctx, const Ptr<Layer>& layer,
                           const std::vector<Arg>& inputs, FuseResult& r);

CV__DNN_INLINE_NS_END
}}

#endif
