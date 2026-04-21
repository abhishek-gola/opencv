// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "graph_fusion_patterns.hpp"
#include <unordered_map>
#include <unordered_set>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// Orchestrates the graph-level fusion pass. The actual pattern matchers live in
// graph_fusion_patterns.{hpp,cpp}; this file only drives them and handles the
// pre-pass that strips no-op BlankLayer (Dropout/Identity) nodes.
struct ModelFusionBasic
{
    explicit ModelFusionBasic(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        eliminateBlankLayers(netimpl->mainGraph);
        netimpl->useCounts(usecounts);
        for (int i = 0; i < 10; i++) {
            if (!fuseGraph(netimpl->mainGraph))
                break;
        }
    }

    // Drop BlankLayer (Dropout/Identity) nodes: they are no-op passthroughs at
    // inference, so we rewrite downstream consumers to read the layer's input
    // directly. Runs before the normal fusion loop so subsequent passes see a
    // smaller graph.
    bool eliminateBlankLayers(Ptr<Graph>& graph)
    {
        if (!graph) return false;
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        const size_t nops = prog.size();

        // Args that are graph outputs must not be renamed away.
        std::unordered_set<int> output_set;
        for (const Arg& a : graph->outputs()) output_set.insert(a.idx);

        // remap[out] = inp — rewrite downstream reads of `out` to read `inp`.
        std::unordered_map<int, Arg> remap;
        std::vector<Ptr<Layer> > newprog;
        newprog.reserve(nops);
        bool modified = false;

        for (size_t i = 0; i < nops; i++) {
            Ptr<Layer> layer = prog[i];
            if (!layer) continue;

            // Apply accumulated remapping to this layer's inputs (chase chains).
            for (Arg& in : layer->inputs) {
                auto it = remap.find(in.idx);
                while (it != remap.end()) {
                    in = it->second;
                    modified = true;
                    it = remap.find(in.idx);
                }
            }

            if (auto* subgraphs = layer->subgraphs()) {
                for (Ptr<Graph>& g : *subgraphs)
                    if (eliminateBlankLayers(g)) modified = true;
            }

            // BlankLayer (Dropout/Identity) can be dropped when it has a single
            // input, no output is a graph output, and any extra outputs (e.g.
            // ONNX Dropout's optional mask) have no downstream uses.
            BlankLayer* blank = dynamic_cast<BlankLayer*>(layer.get());
            if (blank && layer->inputs.size() == 1 && !layer->outputs.empty() &&
                canDropBlank(layer, prog, i, output_set))
            {
                remap[layer->outputs[0].idx] = layer->inputs[0];
                modified = true;
                continue;
            }
            newprog.push_back(layer);
        }

        if (modified) graph->setProg(newprog);
        return modified;
    }

    // Walks the program once, trying each fusion pattern against the current
    // layer. Returns true if any fusion was applied — the outer `fuse()` loop
    // re-runs until the graph reaches a fixed point.
    bool fuseGraph(Ptr<Graph>& graph)
    {
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        const size_t nops = prog.size();
        std::vector<int> producer_of(netimpl->args.size(), -1);
        std::vector<Ptr<Layer> > newprog;
        newprog.reserve(nops);
        bool modified = false;

        FusionContext ctx{netimpl, newprog, producer_of, usecounts};

        for (size_t i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];

            if (auto* subgraphs = layer->subgraphs()) {
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseGraph(g)) modified = true;
            }

            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;

            // Try each pattern in priority order; first match wins.
            FuseResult r;
            bool fused =
                tryFuseTransposeTranspose(ctx, layer, inputs, r) ||
                tryFuseConvBatchNorm     (ctx, layer, inputs, r) ||
                tryFuseConvAddResidual   (ctx, layer, inputs, r) ||
                tryFuseSplitConvConcat   (ctx, layer, inputs, r) ||
                tryFuseParallelConvConcat(ctx, layer, inputs, r) ||
                tryFuseConvActivation    (ctx, layer, inputs, r);

            if (fused) {
                modified = true;
                Layer* surv = newprog[r.layer_idx].get();
                surv->outputs = outputs;
                if (!r.new_inputs.empty()) surv->inputs = r.new_inputs;
                for (const Arg& o : outputs) producer_of[o.idx] = r.layer_idx;
                for (const Arg& a : r.removed_args) {
                    usecounts.at(a.idx) = 0;
                    producer_of.at(a.idx) = -1;
                }
            } else {
                for (const Arg& o : outputs)
                    producer_of[o.idx] = (int)newprog.size();
                newprog.push_back(layer);
            }
        }

        if (modified) {
            // Compact null slots left behind by multi-layer fusions.
            size_t j = 0;
            for (size_t i = 0; i < newprog.size(); i++) {
                if (newprog[i]) {
                    if (j < i) newprog[j] = std::move(newprog[i]);
                    j++;
                }
            }
            newprog.resize(j);
            graph->setProg(newprog);
        }
        return modified;
    }

    // Can we drop a BlankLayer at index `self` in `prog`?  Requires that no
    // output is a graph output and that any extra outputs (e.g. Dropout's mask)
    // are unused by downstream layers.
    bool canDropBlank(const Ptr<Layer>& layer, const std::vector<Ptr<Layer> >& prog,
                      size_t self, const std::unordered_set<int>& output_set) const
    {
        for (const Arg& o : layer->outputs)
            if (output_set.count(o.idx)) return false;
        for (size_t k = 1; k < layer->outputs.size(); k++) {
            const int extra_idx = layer->outputs[k].idx;
            for (size_t m = 0; m < prog.size(); m++) {
                if (!prog[m] || m == self) continue;
                for (const Arg& in : prog[m]->inputs)
                    if (in.idx == extra_idx) return false;
            }
        }
        return true;
    }

    Net::Impl* netimpl;
    std::vector<int> usecounts;
};

void Net::Impl::fuseBasic()
{
    ModelFusionBasic basicFusion(this);
    basicFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}
