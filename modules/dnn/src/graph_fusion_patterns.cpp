// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "graph_fusion_patterns.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace {

template<typename _LayerType>
_LayerType* getLayer(std::vector<Ptr<Layer> >& newprog, int op_idx)
{
    return op_idx >= 0 ? dynamic_cast<_LayerType*>(newprog.at(op_idx).get()) : nullptr;
}

// Every concat input must be produced by a Conv that has usecount==1 and passes `per_conv_ok`.
template<typename Predicate>
bool collectConcatConvs(const FusionContext& ctx, const std::vector<Arg>& inputs,
                        Predicate&& per_conv_ok,
                        std::vector<Conv2Layer*>& convs,
                        std::vector<int>& conv_idx)
{
    const size_t N = inputs.size();
    convs.resize(N);
    conv_idx.resize(N);
    for (size_t k = 0; k < N; k++) {
        int idx = ctx.producer_of.at(inputs[k].idx);
        Conv2Layer* c = getLayer<Conv2Layer>(ctx.newprog, idx);
        if (!c || ctx.usecounts.at(inputs[k].idx) != 1 || !per_conv_ok(c))
            return false;
        convs[k] = c;
        conv_idx[k] = idx;
    }
    return true;
}

// All branches must share dims/type/C_in; weights and (optional) biases must be const.
bool gatherConvWeights(const FusionContext& ctx, const std::vector<Conv2Layer*>& convs,
                       std::vector<Mat>& weights, std::vector<Mat>& biases)
{
    const size_t N = convs.size();
    int ndims_w = -1, C_in = -1, wtype = -1;
    for (size_t k = 0; k < N; k++) {
        Arg w = convs[k]->inputs[1];
        if (!ctx.netimpl->isConstArg(w)) return false;
        weights[k] = ctx.netimpl->argTensor(w);
        if (weights[k].empty() || weights[k].dims < 3) return false;
        if (k == 0) {
            ndims_w = weights[0].dims;
            C_in = weights[0].size[1];
            wtype = weights[0].type();
        } else if (weights[k].dims != ndims_w ||
                   weights[k].size[1] != C_in ||
                   weights[k].type() != wtype) {
            return false;
        }
        if (convs[k]->inputs.size() > 2) {
            Arg b = convs[k]->inputs[2];
            if (!ctx.netimpl->isConstArg(b)) return false;
            biases[k] = ctx.netimpl->argTensor(b);
        }
    }
    return true;
}

// Zero-pads `src` filters into `dst` starting at row `dst_filter_offset`, centering each
// source kernel inside the larger max-kernel footprint. `dst` must be pre-zeroed.
void embedFilterBank(const Mat& src, Mat& dst, int dst_filter_offset,
                     const std::vector<int>& max_k_spatial)
{
    CV_Assert(src.dims == dst.dims && src.dims >= 3);
    const int nspatial = src.dims - 2;
    const int C_in = src.size[1];
    const int oc_k = src.size[0];
    const size_t esz = src.elemSize();

    std::vector<int> pad_lo(nspatial);
    size_t src_spatial = 1, dst_spatial = 1;
    for (int d = 0; d < nspatial; d++)
        pad_lo[d] = (max_k_spatial[d] - src.size[d + 2]) / 2;
    for (int d = nspatial - 1; d >= 0; d--) {
        src_spatial *= src.size[d + 2];
        dst_spatial *= max_k_spatial[d];
    }
    const size_t src_filter_bytes = (size_t)C_in * src_spatial * esz;
    const size_t dst_filter_bytes = (size_t)C_in * dst_spatial * esz;
    const size_t inner_bytes = (size_t)src.size[src.dims - 1] * esz;

    // Innermost dim is a single memcpy; outer spatial dims walk via a multi-index counter.
    const int outer = std::max(nspatial - 1, 0);
    std::vector<int> sidx(outer, 0);

    for (int f = 0; f < oc_k; f++) {
        const uchar* sf = src.ptr() + (size_t)f * src_filter_bytes;
        uchar* df = dst.ptr() + (size_t)(dst_filter_offset + f) * dst_filter_bytes;
        for (int c = 0; c < C_in; c++) {
            const uchar* sc = sf + (size_t)c * src_spatial * esz;
            uchar* dc = df + (size_t)c * dst_spatial * esz;
            std::fill(sidx.begin(), sidx.end(), 0);
            for (;;) {
                size_t s_off = 0, d_off = pad_lo[nspatial - 1];
                size_t sstr = src.size[src.dims - 1];
                size_t dstr = max_k_spatial[nspatial - 1];
                for (int d = nspatial - 2; d >= 0; d--) {
                    s_off += (size_t)sidx[d] * sstr;
                    d_off += (size_t)(pad_lo[d] + sidx[d]) * dstr;
                    sstr *= src.size[d + 2];
                    dstr *= max_k_spatial[d];
                }
                std::memcpy(dc + d_off * esz, sc + s_off * esz, inner_bytes);
                if (outer == 0) break;
                int d = outer - 1;
                while (d >= 0 && ++sidx[d] >= src.size[d + 2]) {
                    sidx[d] = 0;
                    d--;
                }
                if (d < 0) break;
            }
        }
    }
}

void resizeIndexedState(FusionContext& ctx)
{
    const size_t n = ctx.netimpl->args.size();
    ctx.usecounts.resize(n, 0);
    ctx.producer_of.resize(n, -1);
}

} // namespace

bool tryFuseTransposeTranspose(const FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* cur = dynamic_cast<TransposeLayer*>(layer.get());
    if (!cur || inputs.size() != 1 || ctx.usecounts.at(inputs[0].idx) != 1)
        return false;
    const int prev_idx = ctx.producer_of.at(inputs[0].idx);
    auto* prev = getLayer<TransposeLayer>(ctx.newprog, prev_idx);
    if (!prev || prev->perm.empty() || cur->perm.empty() ||
        prev->perm.size() != cur->perm.size())
        return false;

    const size_t ndims = cur->perm.size();
    std::vector<int> composed(ndims);
    for (size_t d = 0; d < ndims; d++) {
        int idx = cur->perm[d];
        if (idx < 0 || idx >= (int)ndims) return false;
        composed[d] = prev->perm[idx];
    }

    bool is_identity = true;
    for (size_t d = 0; d < ndims && is_identity; d++)
        if (composed[d] != (int)d) is_identity = false;

    r.layer_idx = prev_idx;
    if (is_identity) r.new_inputs = prev->inputs;
    else             prev->perm = composed;
    r.removed_args.push_back(inputs[0]);
    return true;
}

bool tryFuseConvBatchNorm(const FusionContext& ctx, const Ptr<Layer>& layer,
                          const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* bn = dynamic_cast<BatchNorm2Layer*>(layer.get());
    if (!bn || inputs.size() != 1 || ctx.usecounts.at(inputs[0].idx) != 1)
        return false;
    const int conv_idx = ctx.producer_of.at(inputs[0].idx);
    auto* conv = getLayer<Conv2Layer>(ctx.newprog, conv_idx);
    if (!conv || !conv->fuseBatchNorm(layer))
        return false;
    r.layer_idx = conv_idx;
    r.removed_args.push_back(inputs[0]);
    return true;
}

bool tryFuseConvAddResidual(const FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* elem = dynamic_cast<NaryEltwiseLayer*>(layer.get());
    if (!elem || inputs.size() != 2) return false;
    if (elem->op != NaryEltwiseLayer::OPERATION::ADD &&
        elem->op != NaryEltwiseLayer::OPERATION::SUM)
        return false;

    const int op0 = ctx.producer_of.at(inputs[0].idx);
    const int op1 = ctx.producer_of.at(inputs[1].idx);
    if (op0 < 0 || op1 < 0) return false;

    // Fold into the later-scheduled operand so the residual is already materialized.
    const int conv_out_i = (op0 > op1) ? 0 : 1;
    const int conv_idx   = (op0 > op1) ? op0 : op1;
    const Arg residual   = inputs[1 - conv_out_i];

    auto* conv = getLayer<Conv2Layer>(ctx.newprog, conv_idx);
    if (!conv || ctx.usecounts.at(inputs[conv_out_i].idx) != 1 ||
        !conv->fuseAddResidual(residual))
        return false;

    r.layer_idx = conv_idx;
    r.removed_args.push_back(inputs[conv_out_i]);
    return true;
}

bool tryFuseSplitConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                            const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* concat = dynamic_cast<Concat2Layer*>(layer.get());
    if (!concat || inputs.size() < 2) return false;
    const size_t N = inputs.size();

    std::vector<Conv2Layer*> convs;
    std::vector<int> conv_idx;
    if (!collectConcatConvs(ctx, inputs,
            [](Conv2Layer* c) { return c->inputs.size() >= 2 && c->ngroups == 1; },
            convs, conv_idx))
        return false;

    const Conv2Layer* c0 = convs[0];
    for (size_t k = 1; k < N; k++) {
        if (convs[k]->strides   != c0->strides ||
            convs[k]->dilations != c0->dilations ||
            convs[k]->pads      != c0->pads)
            return false;
    }

    const int split_idx = ctx.producer_of.at(convs[0]->inputs[0].idx);
    auto* split_layer = getLayer<Split2Layer>(ctx.newprog, split_idx);
    if (!split_layer || split_layer->inputs.size() != 1) return false;
    for (size_t k = 1; k < N; k++) {
        if (ctx.producer_of.at(convs[k]->inputs[0].idx) != split_idx) return false;
    }

    // Require identical kernel shape across branches so weights can be flat-concatenated.
    std::vector<Mat> weights(N), biases(N);
    int outCn_per_group = -1;
    for (size_t k = 0; k < N; k++) {
        Arg w = convs[k]->inputs[1];
        if (!ctx.netimpl->isConstArg(w)) return false;
        weights[k] = ctx.netimpl->argTensor(w);
        if (weights[k].empty() || weights[k].dims < 3) return false;
        if (k == 0) {
            outCn_per_group = weights[0].size[0];
        } else {
            if (weights[k].size[0] != outCn_per_group ||
                weights[k].dims != weights[0].dims ||
                weights[k].type() != weights[0].type())
                return false;
            for (int d = 1; d < weights[k].dims; d++)
                if (weights[k].size[d] != weights[0].size[d]) return false;
        }
        if (convs[k]->inputs.size() > 2) {
            Arg b = convs[k]->inputs[2];
            if (ctx.netimpl->isConstArg(b)) biases[k] = ctx.netimpl->argTensor(b);
        }
    }
    if (outCn_per_group <= 0) return false;

    std::vector<int> mshape(weights[0].dims);
    mshape[0] = outCn_per_group * (int)N;
    for (int d = 1; d < weights[0].dims; d++) mshape[d] = weights[0].size[d];
    Mat merged_w(mshape, weights[0].type());
    const size_t slab_bytes = weights[0].total() * weights[0].elemSize();
    for (size_t k = 0; k < N; k++)
        std::memcpy(merged_w.data + k * slab_bytes, weights[k].data, slab_bytes);

    // Biases are all-or-none; stack per-group along output channels.
    bool have_bias = !biases[0].empty();
    Mat merged_b;
    if (have_bias) {
        const int total_outCn = outCn_per_group * (int)N;
        merged_b.create(1, total_outCn, biases[0].type());
        const size_t esz = biases[0].elemSize();
        for (size_t k = 0; k < N; k++) {
            if (biases[k].empty()) { have_bias = false; break; }
            std::memcpy(merged_b.data + k * outCn_per_group * esz,
                        biases[k].data, outCn_per_group * esz);
        }
    }

    Arg w_arg = ctx.netimpl->newConstArg(convs[0]->name + "_merged_weight", merged_w);
    r.new_inputs = { split_layer->inputs[0], w_arg };
    if (have_bias) {
        Arg b_arg = ctx.netimpl->newConstArg(convs[0]->name + "_merged_bias", merged_b);
        r.new_inputs.push_back(b_arg);
    }
    convs[0]->ngroups = (int)N;
    r.layer_idx = conv_idx[0];
    for (size_t k = 0; k < N; k++) r.removed_args.push_back(inputs[k]);
    for (size_t k = 0; k < N; k++) r.removed_args.push_back(convs[k]->inputs[0]);
    ctx.newprog[split_idx].reset();
    for (size_t k = 1; k < N; k++) ctx.newprog[conv_idx[k]].reset();

    resizeIndexedState(ctx);
    return true;
}

bool tryFuseParallelConvConcat(FusionContext& ctx, const Ptr<Layer>& layer,
                               const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* concat = dynamic_cast<Concat2Layer*>(layer.get());
    if (!concat || inputs.size() < 2 || concat->axis != 1) return false;
    const size_t N = inputs.size();

    std::vector<Conv2Layer*> convs;
    std::vector<int> conv_idx;
    if (!collectConcatConvs(ctx, inputs,
            [](Conv2Layer* c) {
                return c->inputs.size() >= 2 && c->inputs.size() <= 3 &&
                       c->ngroups == 1 && c->auto_pad == AUTO_PAD_NONE;
            },
            convs, conv_idx))
        return false;

    // All branches must share the same input arg (Split-based variant handled elsewhere).
    const int in_idx = convs[0]->inputs[0].idx;
    for (size_t k = 1; k < N; k++)
        if (convs[k]->inputs[0].idx != in_idx) return false;

    const Conv2Layer* c0 = convs[0];
    for (size_t k = 1; k < N; k++) {
        if (convs[k]->strides   != c0->strides ||
            convs[k]->dilations != c0->dilations)
            return false;
    }
    for (size_t k = 1; k < N; k++)
        if (!convs[0]->sameFusedOp(convs[k])) return false;

    std::vector<Mat> weights(N), biases(N);
    if (!gatherConvWeights(ctx, convs, weights, biases)) return false;

    const int ndims_w = weights[0].dims;
    if (ndims_w < 3) return false;
    const int nspatial = ndims_w - 2;
    const int C_in = weights[0].size[1];

    std::vector<int> max_k(nspatial, 0);
    for (int d = 0; d < nspatial; d++)
        for (size_t k = 0; k < N; k++)
            max_k[d] = std::max(max_k[d], weights[k].size[d + 2]);

    // Odd kernels + centered "same" padding keep output spatial shapes aligned across branches.
    const std::vector<int>& dils = convs[0]->dilations;
    for (size_t k = 0; k < N; k++) {
        for (int d = 0; d < nspatial; d++) {
            const int kd = weights[k].size[d + 2];
            if ((kd & 1) == 0) return false;
            const int dilation = dils.empty() ? 1 : dils[d];
            const int expected_pad = ((kd - 1) / 2) * dilation;
            const std::vector<int>& p = convs[k]->pads;
            if (!p.empty() && (int)p.size() != nspatial * 2) return false;
            const int pad_begin = p.empty() ? 0 : p[d];
            const int pad_end   = p.empty() ? 0 : p[d + nspatial];
            if (pad_begin != expected_pad || pad_end != expected_pad) return false;
        }
    }

    bool have_any = false, have_all = true;
    for (size_t k = 0; k < N; k++) {
        if (biases[k].empty()) have_all = false;
        else                   have_any = true;
    }
    if (have_any && !have_all) return false;
    const bool have_bias = have_any;

    int total_outCn = 0;
    for (size_t k = 0; k < N; k++) total_outCn += weights[k].size[0];
    std::vector<int> mshape(ndims_w);
    mshape[0] = total_outCn;
    mshape[1] = C_in;
    for (int d = 0; d < nspatial; d++) mshape[d + 2] = max_k[d];
    Mat merged_w = Mat::zeros(ndims_w, mshape.data(), weights[0].type());

    int out_offset = 0;
    for (size_t k = 0; k < N; k++) {
        embedFilterBank(weights[k], merged_w, out_offset, max_k);
        out_offset += weights[k].size[0];
    }

    Mat merged_b;
    if (have_bias) {
        merged_b.create(1, &total_outCn, biases[0].type());
        const size_t esz = biases[0].elemSize();
        int boff = 0;
        for (size_t k = 0; k < N; k++) {
            const size_t bytes = biases[k].total() * esz;
            std::memcpy(merged_b.data + (size_t)boff * esz, biases[k].data, bytes);
            boff += weights[k].size[0];
        }
    }

    Arg w_arg = ctx.netimpl->newConstArg(convs[0]->name + "_parallel_merged_weight", merged_w);
    r.new_inputs = { convs[0]->inputs[0], w_arg };
    if (have_bias) {
        Arg b_arg = ctx.netimpl->newConstArg(convs[0]->name + "_parallel_merged_bias", merged_b);
        r.new_inputs.push_back(b_arg);
    }
    std::vector<int> new_pads(nspatial * 2);
    for (int d = 0; d < nspatial; d++) {
        const int dilation = dils.empty() ? 1 : dils[d];
        const int p = ((max_k[d] - 1) / 2) * dilation;
        new_pads[d] = new_pads[d + nspatial] = p;
    }
    convs[0]->pads = new_pads;
    r.layer_idx = conv_idx[0];

    for (size_t k = 0; k < N; k++) r.removed_args.push_back(inputs[k]);
    for (size_t k = 1; k < N; k++) ctx.newprog[conv_idx[k]].reset();

    resizeIndexedState(ctx);
    return true;
}

bool tryFuseConvActivation(const FusionContext& ctx, const Ptr<Layer>& layer,
                           const std::vector<Arg>& inputs, FuseResult& r)
{
    auto* activ = dynamic_cast<ActivationLayer*>(layer.get());
    if (!activ || inputs.size() != 1 || ctx.usecounts.at(inputs[0].idx) != 1)
        return false;
    const int conv_idx = ctx.producer_of.at(inputs[0].idx);
    auto* conv = getLayer<Conv2Layer>(ctx.newprog, conv_idx);
    if (!conv || !conv->fuseActivation(layer))
        return false;
    r.layer_idx = conv_idx;
    r.removed_args.push_back(inputs[0]);
    return true;
}

CV__DNN_INLINE_NS_END
}}
