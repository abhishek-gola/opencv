#ifndef OPENCV_DNN_OPDATA_FACTORY_HPP
#define OPENCV_DNN_OPDATA_FACTORY_HPP

#include <opencv2/dnn/dnn.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

typedef Ptr<LayerOpData> (*OpDataConstructor)(
        const LayerParams& params,
        const std::vector<Arg>& inputs,
        const std::vector<Arg>& outputs);

void registerLayerData(const String& type, OpDataConstructor constructor);

Ptr<LayerOpData> createOpData(const String& type,
                              const LayerParams& params,
                              const std::vector<Arg>& inputs,
                              const std::vector<Arg>& outputs);

typedef Ptr<Layer> (*LayerFromDataConstructor)(int backendId, const Ptr<LayerOpData>& data);

void registerLayerFromData(const String& type, int backendId, LayerFromDataConstructor constructor);
Ptr<Layer> createLayerFromData(const String& type, int backendId, const Ptr<LayerOpData>& data);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif  // OPENCV_DNN_OPDATA_FACTORY_HPP
