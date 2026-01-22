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

// Internal registry for typed LayerOpData creation (ENGINE_NEW).
void registerLayerData(const String& type, OpDataConstructor constructor);

// Create a typed LayerOpData instance if registered, otherwise returns a plain LayerOpData.
Ptr<LayerOpData> createOpData(const String& type,
                              const LayerParams& params,
                              const std::vector<Arg>& inputs,
                              const std::vector<Arg>& outputs);

// ENGINE_NEW: registry for backend-specific Layer creation from LayerOpData.
// This allows selecting a backend-specific Layer implementation at Net::finalize() time.
typedef Ptr<Layer> (*LayerFromDataConstructor)(int backendId, const Ptr<LayerOpData>& data);

void registerLayerFromData(const String& type, int backendId, LayerFromDataConstructor constructor);
Ptr<Layer> createLayerFromData(const String& type, int backendId, const Ptr<LayerOpData>& data);

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif  // OPENCV_DNN_OPDATA_FACTORY_HPP
