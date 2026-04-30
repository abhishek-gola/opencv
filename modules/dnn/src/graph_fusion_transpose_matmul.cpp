// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Absorbs a Transpose that only swaps the last two dimensions into the
// trans_a / trans_b parameter of the consuming MatMul. The Transpose is
// dropped when it has a single consumer (the MatMul) and its input has no
// external reference.
//
// Pattern:
//   X -> Transpose(perm=[0,1,..,n-3,n-1,n-2]) -> MatMul(., Y)
//   X -> MatMul(transA=true, ., Y)
//
// (and the symmetric case where the transpose feeds the second operand,
//  which is folded into transB.)
//
// MatMul in cv::dnn treats only the last two axes as the matrix dims and
// broadcasts the leading batch dims, so a perm that leaves all batch axes
// fixed and only flips the trailing two is equivalent to setting transA/transB.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionTransposeMatMul
{
    explicit ModelFusionTransposeMatMul(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    // perm leaves [0..n-3] unchanged and swaps [n-2] with [n-1].
    static bool isLastTwoSwap(const vector<int>& perm)
    {
        int n = (int)perm.size();
        if (n < 2) return false;
        for (int i = 0; i < n - 2; i++)
            if (perm[i] != i) return false;
        return perm[n - 2] == n - 1 && perm[n - 1] == n - 2;
    }

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

            MatMulLayer* mm = dynamic_cast<MatMulLayer*>(layer.get());
            if (!mm) continue;
            // Only handle the two-runtime-input form. Variants with a
            // weight-blob B don't see a Transpose feeding them at runtime.
            if (layer->inputs.size() != 2) continue;

            for (int slot = 0; slot < 2; slot++) {
                Arg in = layer->inputs[slot];
                auto it = producer.find(in.idx);
                if (it == producer.end()) continue;
                int prod_idx = it->second;
                if (prod_idx < 0 || dropped[prod_idx]) continue;

                const Ptr<Layer>& pl = prog[prod_idx];
                TransposeLayer* tr = dynamic_cast<TransposeLayer*>(pl.get());
                if (!tr || pl->outputs.size() != 1) continue;
                if (!isLastTwoSwap(tr->perm)) continue;

                bool single_consumer = usecounts[in.idx] == 1
                                    && externalArgs.count(in.idx) == 0;
                if (!single_consumer) continue;

                // Flip the corresponding trans flag.
                if (slot == 0) mm->trans_a = !mm->trans_a;
                else           mm->trans_b = !mm->trans_b;

                layer->inputs[slot] = pl->inputs[0];
                dropped[prod_idx] = true;
                usecounts[in.idx] = 0;
                modified = true;
            }
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

void Net::Impl::fuseTransposeMatMul()
{
    ModelFusionTransposeMatMul pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
