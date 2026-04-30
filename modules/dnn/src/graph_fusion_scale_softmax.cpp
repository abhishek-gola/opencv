// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Folds a scalar Mul / Div immediately preceding Softmax into the Softmax
// layer's `scale` parameter. The fused softmax bakes the scale into the
// per-element load before max/exp/sum/divide, removing one full pass over
// the (often large) [B, H, S, S] attention-logits tensor.
//
// Pattern:
//   X -> Mul(X, c)  -> Softmax     =>   X -> Softmax(scale=c)
//   X -> Div(X, c)  -> Softmax     =>   X -> Softmax(scale=1/c)
//
// Constraints:
//   - Only fires for the CPU/OpenCV backend; other backends would need their
//     own scaled-softmax kernel to be safe to fuse.
//   - The scalar must be a constant (single-element tensor).
//   - The Mul/Div output must have a single consumer (the Softmax) and not
//     be a graph output, so removing the Mul/Div is safe.
//   - log_softmax is excluded — log(softmax(scale*x)) ≠ scale*log(softmax(x)).

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionScaleSoftmax
{
    explicit ModelFusionScaleSoftmax(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    bool extractScalar(Arg a, float& out) const
    {
        if (!netimpl->isConstArg(a)) return false;
        Mat t = netimpl->argTensor(a);
        if (t.total() != 1) return false;
        if (t.type() == CV_32F) { out = t.ptr<float>()[0]; return true; }
        if (t.type() == CV_64F) { out = (float)t.ptr<double>()[0]; return true; }
        return false;
    }

    void fuse() { fuseGraph(netimpl->mainGraph); }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<Layer>>& prog = graph->prog();
        size_t nops = prog.size();
        bool modified = false;

        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            vector<Ptr<Graph>>* subgraphs = prog[i]->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseGraph(g)) modified = true;
            }
        }

        vector<int> usecounts;
        netimpl->useCounts(usecounts);

        std::set<int> externalArgs;
        for (Arg out : graph->outputs())
            externalArgs.insert(out.idx);

        std::map<int, int> producer;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer[out.idx] = (int)i;
        }

        vector<bool> dropped(nops, false);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            if (!layer || dropped[i]) continue;

            SoftmaxLayer* sm = dynamic_cast<SoftmaxLayer*>(layer.get());
            if (!sm || sm->logSoftMax) continue;
            if (layer->inputs.size() != 1) continue;

            Arg sm_in = layer->inputs[0];
            auto it = producer.find(sm_in.idx);
            if (it == producer.end()) continue;
            int prod_idx = it->second;
            if (prod_idx < 0 || dropped[prod_idx]) continue;

            const Ptr<Layer>& pl = prog[prod_idx];
            NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(pl.get());
            if (!elt) continue;
            const auto op = elt->op;
            const bool is_mul = (op == NaryEltwiseLayer::OPERATION::PROD);
            const bool is_div = (op == NaryEltwiseLayer::OPERATION::DIV);
            if (!is_mul && !is_div) continue;
            if (pl->inputs.size() != 2 || pl->outputs.size() != 1) continue;

            // Identify which operand is the scalar constant.
            float scalar = 0.f;
            int x_slot = -1;
            if (extractScalar(pl->inputs[1], scalar)) {
                x_slot = 0;
            } else if (is_mul && extractScalar(pl->inputs[0], scalar)) {
                // Mul is commutative; Div with the data on the right doesn't
                // simplify (1 / scale*x ≠ softmax-fold) so only handle Mul here.
                x_slot = 1;
            } else {
                continue;
            }

            if (is_div) {
                if (scalar == 0.f) continue;
                scalar = 1.f / scalar;
            }
            if (!std::isfinite(scalar)) continue;

            bool single_consumer = usecounts[sm_in.idx] == 1
                                && externalArgs.count(sm_in.idx) == 0;
            if (!single_consumer) continue;

            // Bake the scalar into the Softmax. Compose with any pre-existing
            // scale (defaults to 1) so multiple passes can keep accumulating.
            sm->scale *= scalar;
            layer->inputs[0] = pl->inputs[x_slot];
            dropped[prod_idx] = true;
            usecounts[sm_in.idx] = 0;
            modified = true;
        }

        if (modified) {
            vector<Ptr<Layer>> newprog;
            newprog.reserve(nops);
            for (size_t i = 0; i < nops; i++) {
                if (!dropped[i] && prog[i])
                    newprog.push_back(prog[i]);
            }
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseScaleSoftmax()
{
    // The scale path is implemented for the OpenCV (CPU) backend's Softmax
    // kernel only. For other backends, leave the Mul/Div + Softmax pair
    // untouched so the runtime stays correct.
    if (preferableBackend != DNN_BACKEND_OPENCV ||
        preferableTarget  != DNN_TARGET_CPU)
        return;
    ModelFusionScaleSoftmax pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
