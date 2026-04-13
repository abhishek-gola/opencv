// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

struct ModelFusionBasic
{
    ModelFusionBasic(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        int i, niter = 10;
        netimpl->useCounts(usecounts);
        for (i = 0; i < niter; i++) {
            bool fused_any = fuseGraph(netimpl->mainGraph);
            if (!fused_any)
                break;
        }
    }

    template<typename _LayerType> _LayerType*
    getLayer(std::vector<Ptr<Layer> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? dynamic_cast<_LayerType*>(newprog.at(op_idx).get()) : 0;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        vector<Arg> removed_args;
        bool modified = false;
        const std::vector<Ptr<Layer> >& prog = graph->prog();
        size_t i, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> producer_of(nargs, -1);
        std::vector<Ptr<Layer> > newprog;
        std::vector<Arg> fused_inputs;

        for (i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            Layer* layer_ptr = (Layer*)layer.get();
            int fused_layer_idx = -1;
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    if (fuseGraph(g))
                        modified = true;
                }
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t ninputs = inputs.size();
            removed_args.clear();
            fused_inputs.clear(); // leave it empty in the merge patterns below to re-use original fused node inputs as-is.

            for(;;) {
                BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(layer_ptr);
                ActivationLayer* activ = dynamic_cast<ActivationLayer*>(layer_ptr);
                NaryEltwiseLayer* elemwise = dynamic_cast<NaryEltwiseLayer*>(layer_ptr);
                TransposeLayer* transpose = dynamic_cast<TransposeLayer*>(layer_ptr);

                // merge two consecutive transposes into one:
                // Transpose(perm1) -> Transpose(perm2) => Transpose(composed_perm)
                if (transpose && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg tr_inp = inputs[0];
                    int prev_layer_idx = producer_of.at(tr_inp.idx);
                    TransposeLayer* prev_transpose = getLayer<TransposeLayer>(newprog, prev_layer_idx);
                    if (prev_transpose && !prev_transpose->perm.empty() && !transpose->perm.empty() &&
                        prev_transpose->perm.size() == transpose->perm.size()) {
                        size_t ndims = transpose->perm.size();
                        // composed_perm[i] = prev_perm[cur_perm[i]]
                        std::vector<int> composed_perm(ndims);
                        bool valid = true;
                        for (size_t d = 0; d < ndims; d++) {
                            int idx = transpose->perm[d];
                            if (idx < 0 || idx >= (int)ndims) { valid = false; break; }
                            composed_perm[d] = prev_transpose->perm[idx];
                        }
                        if (valid) {
                            // check if composed perm is identity
                            bool is_identity = true;
                            for (size_t d = 0; d < ndims; d++) {
                                if (composed_perm[d] != (int)d) { is_identity = false; break; }
                            }
                            if (is_identity) {
                                // the two transposes cancel each other out;
                                // rewire: prev_transpose's input feeds directly to our output
                                fused_layer_idx = prev_layer_idx;
                                fused_inputs = prev_transpose->inputs;
                                removed_args.push_back(tr_inp);
                            } else {
                                prev_transpose->perm = composed_perm;
                                fused_layer_idx = prev_layer_idx;
                                removed_args.push_back(tr_inp);
                            }
                            break;
                        }
                    }
                }

                // merge convolution and batch norm
                if (bn && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg bn_inp = inputs[0];
                    int conv_layer_idx = producer_of.at(bn_inp.idx);
                    Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                    if (conv) {
                        bool ok = conv->fuseBatchNorm(layer);
                        if (ok) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(bn_inp);
                            break;
                        }
                    }
                }

                // merge residual 'add' into 'conv' node
                if (elemwise && (elemwise->op == NaryEltwiseLayer::OPERATION::ADD ||
                    elemwise->op == NaryEltwiseLayer::OPERATION::SUM) &&
                    ninputs == 2) {

                    int op0 = producer_of.at(inputs[0].idx);
                    int op1 = producer_of.at(inputs[1].idx);

                    if (op0 >= 0 && op1 >= 0) {
                        int conv_layer_idx;
                        Arg residual, conv_out;

                        if (op0 > op1) { // choose the latter op to ensure that the other component is already computed
                            conv_layer_idx = op0;
                            conv_out = inputs[0];
                            residual = inputs[1];
                        } else {
                            conv_layer_idx = op1;
                            conv_out = inputs[1];
                            residual = inputs[0];
                        }

                        Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                        if (conv && usecounts[conv_out.idx] == 1 &&
                            conv->fuseAddResidual(residual)) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(conv_out);
                            break;
                        }
                    }
                }

                // merge split => multiple convs => concat into a single grouped convolution
                {
                    Concat2Layer* concat = dynamic_cast<Concat2Layer*>(layer_ptr);
                    if (concat && ninputs >= 2) {
                        // check that all inputs come from Conv2Layers
                        bool all_convs = true;
                        std::vector<int> conv_indices(ninputs);
                        std::vector<Conv2Layer*> convs(ninputs);
                        for (size_t k = 0; k < ninputs; k++) {
                            int idx = producer_of.at(inputs[k].idx);
                            Conv2Layer* c = getLayer<Conv2Layer>(newprog, idx);
                            if (!c || c->inputs.size() < 2 || c->ngroups != 1 ||
                                usecounts.at(inputs[k].idx) != 1) {
                                all_convs = false;
                                break;
                            }
                            conv_indices[k] = idx;
                            convs[k] = c;
                        }
                        if (all_convs) {
                            // check that all convs have the same params
                            const Conv2Layer* c0 = convs[0];
                            bool same_params = true;
                            for (size_t k = 1; k < ninputs && same_params; k++) {
                                const Conv2Layer* ck = convs[k];
                                if (ck->strides != c0->strides ||
                                    ck->dilations != c0->dilations ||
                                    ck->pads != c0->pads)
                                    same_params = false;
                            }
                            if (same_params) {
                                // check that all convs take input from the same Split2 layer
                                int split_idx = -1;
                                Split2Layer* split_layer = nullptr;
                                bool from_same_split = true;
                                for (size_t k = 0; k < ninputs && from_same_split; k++) {
                                    Arg conv_data = convs[k]->inputs[0];
                                    int pidx = producer_of.at(conv_data.idx);
                                    Split2Layer* sp = getLayer<Split2Layer>(newprog, pidx);
                                    if (!sp) { from_same_split = false; break; }
                                    if (k == 0) { split_idx = pidx; split_layer = sp; }
                                    else if (pidx != split_idx) { from_same_split = false; }
                                }
                                if (from_same_split && split_layer && split_layer->inputs.size() == 1) {
                                    // check that all conv weights are constant and have compatible shapes
                                    bool weights_ok = true;
                                    std::vector<Mat> weight_mats(ninputs);
                                    std::vector<Mat> bias_mats(ninputs);
                                    int outCn_per_group = -1;
                                    for (size_t k = 0; k < ninputs && weights_ok; k++) {
                                        Arg w_arg = convs[k]->inputs[1];
                                        if (!netimpl->isConstArg(w_arg)) { weights_ok = false; break; }
                                        weight_mats[k] = netimpl->argTensor(w_arg);
                                        if (weight_mats[k].empty() || weight_mats[k].dims < 3) { weights_ok = false; break; }
                                        int oc = weight_mats[k].size[0];
                                        if (k == 0) outCn_per_group = oc;
                                        else if (oc != outCn_per_group) { weights_ok = false; break; }
                                        // check kernel dims match
                                        if (k > 0 && weight_mats[k].dims != weight_mats[0].dims) { weights_ok = false; break; }
                                        for (int d = 2; d < weight_mats[k].dims && weights_ok; d++) {
                                            if (weight_mats[k].size[d] != weight_mats[0].size[d])
                                                weights_ok = false;
                                        }
                                        if (convs[k]->inputs.size() > 2) {
                                            Arg b_arg = convs[k]->inputs[2];
                                            if (netimpl->isConstArg(b_arg))
                                                bias_mats[k] = netimpl->argTensor(b_arg);
                                        }
                                    }
                                    if (weights_ok && outCn_per_group > 0) {
                                        // concatenate weights along axis 0
                                        int ndims_w = weight_mats[0].dims;
                                        std::vector<int> merged_wshape(ndims_w);
                                        merged_wshape[0] = outCn_per_group * (int)ninputs;
                                        for (int d = 1; d < ndims_w; d++)
                                            merged_wshape[d] = weight_mats[0].size[d];
                                        Mat merged_weights(merged_wshape, weight_mats[0].type());
                                        int slice_size = (int)(weight_mats[0].total() / outCn_per_group) * outCn_per_group;
                                        for (size_t k = 0; k < ninputs; k++) {
                                            size_t bytes = weight_mats[k].total() * weight_mats[k].elemSize();
                                            memcpy(merged_weights.data + k * bytes, weight_mats[k].data, bytes);
                                        }

                                        // concatenate biases
                                        int total_outCn = outCn_per_group * (int)ninputs;
                                        bool have_bias = !bias_mats[0].empty();
                                        Mat merged_bias;
                                        if (have_bias) {
                                            merged_bias.create(1, total_outCn, bias_mats[0].type());
                                            size_t bias_esz = bias_mats[0].elemSize();
                                            for (size_t k = 0; k < ninputs; k++) {
                                                if (bias_mats[k].empty()) { have_bias = false; break; }
                                                memcpy(merged_bias.data + k * outCn_per_group * bias_esz,
                                                       bias_mats[k].data, outCn_per_group * bias_esz);
                                            }
                                        }

                                        if (weights_ok) {
                                            // create merged weight and bias args
                                            Arg merged_w_arg = netimpl->newConstArg(
                                                convs[0]->name + "_merged_weight", merged_weights);
                                            std::vector<Arg> new_inputs = {split_layer->inputs[0], merged_w_arg};
                                            if (have_bias) {
                                                Arg merged_b_arg = netimpl->newConstArg(
                                                    convs[0]->name + "_merged_bias", merged_bias);
                                                new_inputs.push_back(merged_b_arg);
                                            }

                                            // update the first conv to be a grouped conv
                                            Conv2Layer* merged_conv = convs[0];
                                            merged_conv->ngroups = (int)ninputs;
                                            fused_layer_idx = conv_indices[0];
                                            fused_inputs = new_inputs;

                                            // mark intermediate args as removed
                                            for (size_t k = 0; k < ninputs; k++)
                                                removed_args.push_back(inputs[k]); // conv outputs
                                            for (size_t k = 0; k < ninputs; k++)
                                                removed_args.push_back(convs[k]->inputs[0]); // split outputs

                                            // remove the split and other conv layers
                                            newprog[split_idx] = Ptr<Layer>();
                                            for (size_t k = 1; k < ninputs; k++)
                                                newprog[conv_indices[k]] = Ptr<Layer>();

                                            // update usecounts and producer_of for new args
                                            size_t new_nargs = netimpl->args.size();
                                            usecounts.resize(new_nargs, 0);
                                            producer_of.resize(new_nargs, -1);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // merge convolution and activation
                if (activ && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg activ_inp = inputs[0];
                    int conv_layer_idx = producer_of.at(activ_inp.idx);
                    Conv2Layer* conv = getLayer<Conv2Layer>(newprog, conv_layer_idx);
                    if (conv) {
                        bool ok = conv->fuseActivation(layer);
                        if (ok) {
                            fused_layer_idx = conv_layer_idx;
                            removed_args.push_back(activ_inp);
                            break;
                        }
                    }
                }
                break;
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = newprog[fused_layer_idx];
                fused_layer->outputs = outputs;
                for (Arg new_out: outputs)
                    producer_of[new_out.idx] = fused_layer_idx;
                for (Arg old_out: removed_args) {
                    usecounts.at(old_out.idx) = 0;
                    producer_of.at(old_out.idx) = -1;
                }
            } else {
                for (auto out: outputs)
                    producer_of[out.idx] = (int)newprog.size();
                newprog.push_back(layer);
            }
        }

        if (modified) {
            size_t i, j = 0, newops = newprog.size();
            for (i = 0; i < newops; i++) {
                if (!newprog[i].empty()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            //printf("fused some ops in graph %s. size before: %zu ops, size after: %zu ops\n",
            //       graph->name().data(), nops, j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;
    vector<int> usecounts;
};

void Net::Impl::fuseBasic()
{
    ModelFusionBasic basicFusion(this);
    basicFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}
