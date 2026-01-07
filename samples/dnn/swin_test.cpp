#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    // Default ONNX path matches your Python export
    std::string onnxPath = "swin_t_repro.onnx";
    if (argc >= 2) onnxPath = argv[1];

    std::cout << "OpenCV version: " << CV_VERSION << "\n";
    std::cout << "Loading ONNX into OpenCV DNN...\n";
    std::cout << "ONNX path: " << onnxPath << "\n";

    try {
        // This is the line that reproduces the crash/issue in many cases
        cv::dnn::Net net = cv::dnn::readNet(onnxPath);

        // Optional: print something so you know it got past readNet
        std::cout << "Successfully loaded net.\n";
        std::cout << "Empty? " << (net.empty() ? "yes" : "no") << "\n";
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception:\n" << e.what() << "\n";
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "std::exception:\n" << e.what() << "\n";
        return 3;
    }

    return 0;
}
