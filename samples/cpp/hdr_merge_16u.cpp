// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>

static void printUsage(const char* argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " <algo> <out_prefix> <t1> <img1> [<t2> <img2> ...]\n"
        << "  " << argv0 << " --iisc070 [debevec|robertson]\n"
        << "\n"
        << "  <algo>       : debevec | robertson\n"
        << "  <out_prefix> : output prefix for 16-bit PNG(s)\n"
        << "  <ti>         : exposure time in seconds (float)\n"
        << "  <imgi>       : path to exposure image (16-bit TIFF recommended)\n"
        << "\n"
        << "Example:\n"
        << "  " << argv0 << " debevec out 0.033 input_1.tif 0.25 input_2.tif 1.0 input_3.tif\n";
}

static bool loadIISc070(std::vector<cv::Mat>& images, std::vector<float>& times)
{
    const std::string dir = "/home/abhishek/Downloads/hdr_images/IISc_VAL_HDRRNN_dataset/training_data/Input_070";
    const std::string expPath = dir + "/input_exp.txt";

    std::ifstream f(expPath.c_str());
    if (!f.is_open())
        return false;

    std::vector<float> bias;
    float b = 0.f;
    while (f >> b)
        bias.push_back(b);

    if (bias.size() != 7)
        return false;

    images.clear();
    times.clear();
    images.reserve(7);
    times.reserve(7);

    for (int i = 1; i <= 7; ++i)
    {
        const std::string fn = dir + "/input_" + std::to_string(i) + "_aligned.tif";
        cv::Mat img = cv::imread(fn, cv::IMREAD_UNCHANGED);
        if (img.empty())
            return false;
        images.push_back(img);
    }

    // Dataset README: exposure bias values.
    for (float v : bias)
        times.push_back(std::pow(2.0f, v));

    return true;
}

static double computeLuminanceMax(const cv::Mat& hdr)
{
    CV_Assert(hdr.type() == CV_32FC3);

    std::vector<cv::Mat> ch;
    cv::split(hdr, ch);

    cv::Mat lum = 0.2126f * ch[2] + 0.7152f * ch[1] + 0.0722f * ch[0];

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(lum, &minv, &maxv);

    if (!(maxv > 0.0))
        maxv = 1.0;

    return maxv;
}

static bool saveRawHDR16Png(const cv::Mat& hdr32f, const std::string& pngPath, cv::Mat* out16u = nullptr)
{
    CV_Assert(hdr32f.type() == CV_32FC3);

    double maxLum = computeLuminanceMax(hdr32f);
    cv::Mat norm = hdr32f * (1.0 / maxLum);

    cv::pow(norm, 1.0f / 2.2f, norm);

    cv::Mat png16u;
    norm.convertTo(png16u, CV_16U, 65535.0);

    if (out16u)
        *out16u = png16u;
    return cv::imwrite(pngPath, png16u);
}

static bool saveReinhard16Png(const cv::Mat& hdr32f, const std::string& pngPath, cv::Mat* out16u = nullptr)
{
    CV_Assert(hdr32f.type() == CV_32FC3);

    double maxLum = computeLuminanceMax(hdr32f);
    cv::Mat hdrNorm = hdr32f * (1.0 / maxLum);

    cv::Ptr<cv::TonemapReinhard> tm = cv::createTonemapReinhard(1.5f, 0.f, 0.f, 0.f);

    cv::Mat ldr32f;
    tm->process(hdrNorm, ldr32f);

    cv::threshold(ldr32f, ldr32f, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(ldr32f, ldr32f, 1.0, 1.0, cv::THRESH_TRUNC);

    cv::Mat png16u;
    ldr32f.convertTo(png16u, CV_16U, 65535.0);
    if (out16u)
        *out16u = png16u;
    return cv::imwrite(pngPath, png16u);
}

int main(int argc, char** argv)
{
    if (argc == 1 || (argc >= 2 && std::string(argv[1]) == "--iisc070"))
    {
        const std::string algo = (argc >= 3) ? argv[2] : "debevec";
        const std::string outPrefix = std::string("out_iisc_070_") + algo;

        std::vector<cv::Mat> images;
        std::vector<float> times;
        if (!loadIISc070(images, times))
        {
            std::cerr << "Couldn't load IISc Input_070 default dataset.\n";
            printUsage(argv[0]);
            return 1;
        }

        cv::Mat hdr;
        if (algo == "debevec")
        {
            cv::Ptr<cv::MergeDebevec> merge = cv::createMergeDebevec();
            merge->process(images, hdr, times);
        }
        else if (algo == "robertson")
        {
            cv::Ptr<cv::MergeRobertson> merge = cv::createMergeRobertson();
            merge->process(images, hdr, times);
        }
        else
        {
            std::cerr << "Unknown algo: " << algo << "\n";
            return 2;
        }

        std::cout << "Loaded " << images.size() << " exposures:\n";
        for (size_t i = 0; i < images.size(); ++i)
            std::cout << "  [" << i << "] time=" << times[i] << " depth=" << images[i].depth() << " channels=" << images[i].channels() << "\n";

        const std::string pngRawPath = outPrefix + "_raw16.png";
        const std::string pngTmPath  = outPrefix + "_reinhard16.png";

        cv::Mat pngRaw16u, pngTm16u;
        if (!saveRawHDR16Png(hdr, pngRawPath, &pngRaw16u))
        {
            std::cerr << "Failed to write: " << pngRawPath << "\n";
            return 3;
        }
        if (!saveReinhard16Png(hdr, pngTmPath, &pngTm16u))
        {
            std::cerr << "Failed to write: " << pngTmPath << "\n";
            return 4;
        }

        std::cout << "Wrote: " << pngRawPath << "\n";
        std::cout << "Wrote: " << pngTmPath << "\n";

        {
            cv::Mat check = cv::imread(pngTmPath, cv::IMREAD_UNCHANGED);
            std::cout << "PNG depth = " << check.depth() << " channels = " << check.channels() << "\n";
        }

        try
        {
            cv::namedWindow("preview", cv::WINDOW_NORMAL);
            cv::Mat preview8u;
            pngTm16u.convertTo(preview8u, CV_8U, 1.0 / 257.0);
            cv::imshow("preview", preview8u);
            std::cout << "Press any key in the image window to exit.\n";
            cv::waitKey();
        }
        catch (...)
        {
        }

        return 0;
    }

    if (argc < 5 || ((argc - 3) % 2) != 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string algo = argv[1];
    const std::string outPrefix = argv[2];

    std::vector<cv::Mat> images;
    std::vector<float> times;
    images.reserve((argc - 3) / 2);
    times.reserve((argc - 3) / 2);

    for (int i = 3; i < argc; i += 2)
    {
        const float t = std::stof(argv[i]);
        const std::string path = argv[i + 1];
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            std::cerr << "Can't read image: " << path << "\n";
            return 2;
        }
        if (img.depth() != CV_16U && img.depth() != CV_8U && img.depth() != CV_32F)
        {
            std::cerr << "Unsupported depth for " << path << " (depth=" << img.depth() << ").\n";
            return 3;
        }
        if (img.channels() != 1 && img.channels() != 3)
        {
            std::cerr << "Unsupported channels for " << path << " (channels=" << img.channels() << ").\n";
            return 4;
        }

        images.push_back(img);
        times.push_back(t);
    }

    cv::Mat hdr;
    if (algo == "debevec")
    {
        cv::Ptr<cv::MergeDebevec> merge = cv::createMergeDebevec();
        merge->process(images, hdr, times);
    }
    else if (algo == "robertson")
    {
        cv::Ptr<cv::MergeRobertson> merge = cv::createMergeRobertson();
        merge->process(images, hdr, times);
    }
    else
    {
        std::cerr << "Unknown algo: " << algo << "\n";
        printUsage(argv[0]);
        return 5;
    }

    const std::string pngRawPath = outPrefix + "_raw16.png";
    const std::string pngTmPath  = outPrefix + "_reinhard16.png";
    cv::Mat pngRaw16u, pngTm16u;
    if (!saveRawHDR16Png(hdr, pngRawPath, &pngRaw16u))
    {
        std::cerr << "Failed to write: " << pngRawPath << "\n";
        return 6;
    }
    if (!saveReinhard16Png(hdr, pngTmPath, &pngTm16u))
    {
        std::cerr << "Failed to write: " << pngTmPath << "\n";
        return 7;
    }

    std::cout << "Wrote: " << pngRawPath << "\n";
    std::cout << "Wrote: " << pngTmPath << "\n";

    {
        cv::Mat check = cv::imread(pngTmPath, cv::IMREAD_UNCHANGED);
        std::cout << "PNG depth = " << check.depth() << " channels = " << check.channels() << "\n";
    }

    try
    {
        cv::namedWindow("preview", cv::WINDOW_NORMAL);
        cv::Mat preview8u;
        pngTm16u.convertTo(preview8u, CV_8U, 1.0 / 257.0);
        cv::imshow("preview", preview8u);
        std::cout << "Press any key in the image window to exit.\n";
        cv::waitKey();
    }
    catch (...)
    {
        // HighGUI may be unavailable in some builds; writing files is enough.
    }

    return 0;
}
