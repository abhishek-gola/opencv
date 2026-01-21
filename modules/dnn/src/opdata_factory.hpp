#ifndef OPENCV_DNN_OPDATA_FACTORY_HPP
#define OPENCV_DNN_OPDATA_FACTORY_HPP

#include <opencv2/dnn/dnn.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

typedef Ptr<OpData> (*OpDataConstructor)(
        const LayerParams& params,
        const std::vector<Arg>& inputs,
        const std::vector<Arg>& outputs);

// Internal registry for typed OpData creation (ENGINE_NEW / AbstractGraph).
void registerOpData(const String& type, OpDataConstructor constructor);

// Create a typed OpData instance if registered, otherwise returns a plain OpData.
Ptr<OpData> createOpData(const String& type,
                         const LayerParams& params,
                         const std::vector<Arg>& inputs,
                         const std::vector<Arg>& outputs);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif  // OPENCV_DNN_OPDATA_FACTORY_HPP
