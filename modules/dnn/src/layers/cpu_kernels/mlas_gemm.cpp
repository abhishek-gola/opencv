// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "mlas_gemm.hpp"

#ifdef HAVE_MLAS

// Vendored MLAS public header. Keep this include narrow (only the public API
// — no mlasi.h, no platform detection internals).
#include "mlas.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TEMPORARY (verification): emit a one-line stderr report at process exit
// counting how many times mlasSgemm / mlasSgemmBatch were called. Set
// OPENCV_DNN_MLAS_TRACE=0 in the env to silence.
namespace {
struct MlasCallCounter {
    std::atomic<size_t> single{0};
    std::atomic<size_t> batched{0};
    std::atomic<size_t> single_skipped{0};
    ~MlasCallCounter() {
        const char* e = std::getenv("OPENCV_DNN_MLAS_TRACE");
        if (e && std::string(e) == "0") return;
        std::fprintf(stderr,
            "[MLAS] mlasSgemm calls: %zu, mlasSgemmBatch calls: %zu, "
            "skipped (unavailable/strided): %zu\n",
            single.load(), batched.load(), single_skipped.load());
    }
};
MlasCallCounter g_mlas_counter;
}  // namespace

namespace cv { namespace dnn {

bool mlasAvailable() {
    // MlasGetPreferredBufferAlignment() is the cheapest non-stateful call we
    // can make to confirm MLAS initialised its dispatch table without
    // crashing on this host. It returns a small power-of-two on every CPU
    // MLAS supports.
    static const bool ok = []() {
        const size_t a = MlasGetPreferredBufferAlignment();
        return a > 0 && a <= 256;
    }();
    return ok;
}

bool mlasSgemm(bool trans_a, bool trans_b,
               int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc)
{
    if (!mlasAvailable()) { g_mlas_counter.single_skipped++; return false; }
    if (M <= 0 || N <= 0 || K <= 0) { g_mlas_counter.single_skipped++; return false; }

    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = static_cast<size_t>(lda);
    data.B = B;
    data.ldb = static_cast<size_t>(ldb);
    data.C = C;
    data.ldc = static_cast<size_t>(ldc);
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = false;

    MlasGemm(trans_a ? CblasTrans : CblasNoTrans,
             trans_b ? CblasTrans : CblasNoTrans,
             static_cast<size_t>(M),
             static_cast<size_t>(N),
             static_cast<size_t>(K),
             data,
             /*ThreadPool=*/nullptr,
             /*BackendKernelSelectorConfig=*/nullptr);
    g_mlas_counter.single++;
    return true;
}

bool mlasSgemmBatch(size_t batch,
                    const size_t* A_offsets,
                    const size_t* B_offsets,
                    const size_t* C_offsets,
                    bool trans_a, bool trans_b,
                    int M, int N, int K,
                    float alpha,
                    const float* A_base, int lda,
                    const float* B_base, int ldb,
                    float beta,
                    float* C_base, int ldc)
{
    if (!mlasAvailable()) return false;
    if (batch == 0 || M <= 0 || N <= 0 || K <= 0) return false;

    std::vector<MLAS_SGEMM_DATA_PARAMS> data(batch);
    for (size_t i = 0; i < batch; i++) {
        data[i].A = A_base + A_offsets[i];
        data[i].lda = static_cast<size_t>(lda);
        data[i].B = B_base + B_offsets[i];
        data[i].ldb = static_cast<size_t>(ldb);
        data[i].C = C_base + C_offsets[i];
        data[i].ldc = static_cast<size_t>(ldc);
        data[i].alpha = alpha;
        data[i].beta = beta;
        data[i].BIsPacked = false;
    }

    MlasGemmBatch(trans_a ? CblasTrans : CblasNoTrans,
                  trans_b ? CblasTrans : CblasNoTrans,
                  static_cast<size_t>(M),
                  static_cast<size_t>(N),
                  static_cast<size_t>(K),
                  data.data(),
                  batch,
                  /*ThreadPool=*/nullptr,
                  /*BackendKernelSelectorConfig=*/nullptr);
    g_mlas_counter.batched += batch;
    return true;
}

size_t mlasSgemmPackBSize(bool trans_a, bool trans_b, int N, int K)
{
    if (!mlasAvailable()) return 0;
    if (N <= 0 || K <= 0) return 0;
    return MlasGemmPackBSize(trans_a ? CblasTrans : CblasNoTrans,
                             trans_b ? CblasTrans : CblasNoTrans,
                             static_cast<size_t>(N),
                             static_cast<size_t>(K),
                             /*BackendKernelSelectorConfig=*/nullptr);
}

bool mlasSgemmPackB(bool trans_a, bool trans_b, int N, int K,
                    const float* B, int ldb, void* packed_B)
{
    if (!mlasAvailable()) return false;
    if (N <= 0 || K <= 0 || B == nullptr || packed_B == nullptr) return false;
    MlasGemmPackB(trans_a ? CblasTrans : CblasNoTrans,
                  trans_b ? CblasTrans : CblasNoTrans,
                  static_cast<size_t>(N),
                  static_cast<size_t>(K),
                  B, static_cast<size_t>(ldb),
                  packed_B,
                  /*BackendKernelSelectorConfig=*/nullptr);
    return true;
}

bool mlasSgemmPacked(bool trans_a, bool trans_b,
                     int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const void* packed_B,
                     float beta,
                     float* C, int ldc)
{
    if (!mlasAvailable()) { g_mlas_counter.single_skipped++; return false; }
    if (M <= 0 || N <= 0 || K <= 0) { g_mlas_counter.single_skipped++; return false; }

    MLAS_SGEMM_DATA_PARAMS data;
    data.A = A;
    data.lda = static_cast<size_t>(lda);
    data.B = static_cast<const float*>(packed_B);
    data.ldb = 0;  // ignored when BIsPacked
    data.C = C;
    data.ldc = static_cast<size_t>(ldc);
    data.alpha = alpha;
    data.beta = beta;
    data.BIsPacked = true;

    MlasGemm(trans_a ? CblasTrans : CblasNoTrans,
             trans_b ? CblasTrans : CblasNoTrans,
             static_cast<size_t>(M),
             static_cast<size_t>(N),
             static_cast<size_t>(K),
             data,
             /*ThreadPool=*/nullptr,
             /*BackendKernelSelectorConfig=*/nullptr);
    g_mlas_counter.single++;
    return true;
}

}}  // cv::dnn

#endif  // HAVE_MLAS
