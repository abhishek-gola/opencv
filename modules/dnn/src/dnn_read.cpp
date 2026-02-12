// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net readNet(const String& _model, const String& _config, const String& _framework, int engine)
{
    String framework = toLowerCase(_framework);
    String model = _model;
    String config = _config;
    const std::string modelExt = model.substr(model.rfind('.') + 1);
    const std::string configExt = config.substr(config.rfind('.') + 1);
    if (framework == "tflite" || modelExt == "tflite")
    {
        return readNetFromTFLite(model, engine);
    }
    if (framework == "onnx" || modelExt == "onnx")
    {
        return readNetFromONNX(model, engine);
    }
    CV_Error(Error::StsError, "Cannot determine an origin framework of files: " + model + (config.empty() ? "" : ", " + config));
}

Net readNet(const String& _framework, const std::vector<uchar>& bufferModel,
        const std::vector<uchar>& bufferConfig, int engine)
{
    String framework = toLowerCase(_framework);
    if (framework == "onnx")
        return readNetFromONNX(bufferModel, engine);
    else if (framework == "tflite")
        return readNetFromTFLite(bufferModel, engine);
    CV_Error(Error::StsError, "Cannot determine an origin framework with a name " + framework);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
