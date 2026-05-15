// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "mlas_gemm.hpp"

#ifdef HAVE_MLAS

#include "mlas.h"
#include <opencv2/core/utility.hpp>
#include <vector>

// MLAS_SGEMM_THREAD_COMPLEXITY is private to lib/mlasi.h, but its value is
// fixed by the vendored SGEMM dispatch logic. Keep this in sync.
#define MLAS_OPENCV_SGEMM_THREAD_COMPLEXITY (size_t(64) * size_t(1024))

namespace cv { namespace dnn {

bool mlasAvailable() {
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
    if (!mlasAvailable()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;

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

    const CBLAS_TRANSPOSE tA = trans_a ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE tB = trans_b ? CblasTrans : CblasNoTrans;
    const size_t Ms = static_cast<size_t>(M);
    const size_t Ns = static_cast<size_t>(N);
    const size_t Ks = static_cast<size_t>(K);

    // MLAS's MlasGemmBatch splits its TargetThreadCount across `batch`
    // gemms (ThreadsPerGemm = TargetThreadCount / batch), so when each
    // gemm is already large enough to saturate every worker on its own,
    // batching gives ThreadsPerGemm = 1..few — fewer M-rows per thread,
    // but also each thread now juggles a *different* B matrix per batch
    // slot, killing L1/L2 reuse of the packed B panel.
    //
    // Process the batch sequentially in that regime: one gemm fans out to
    // all threads, threads share the packed B panel, then move on. The
    // attention QK^T / scores*V matmuls in ViT/CLIP backbones are the hot
    // case here (M=N=seq, K=head_dim, batch=heads).
    const size_t per_gemm = Ms * Ns * Ks;
    const int nt = std::max(1, cv::getNumThreads());
    const size_t saturate = MLAS_OPENCV_SGEMM_THREAD_COMPLEXITY *
                            static_cast<size_t>(nt);
    if (batch > 1 && per_gemm >= saturate) {
        MLAS_SGEMM_DATA_PARAMS one;
        one.lda = static_cast<size_t>(lda);
        one.ldb = static_cast<size_t>(ldb);
        one.ldc = static_cast<size_t>(ldc);
        one.alpha = alpha;
        one.beta = beta;
        one.BIsPacked = false;
        for (size_t i = 0; i < batch; i++) {
            one.A = A_base + A_offsets[i];
            one.B = B_base + B_offsets[i];
            one.C = C_base + C_offsets[i];
            MlasGemmBatch(tA, tB, Ms, Ns, Ks, &one, 1,
                          /*ThreadPool=*/nullptr,
                          /*BackendKernelSelectorConfig=*/nullptr);
        }
        return true;
    }

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

    MlasGemmBatch(tA, tB, Ms, Ns, Ks,
                  data.data(),
                  batch,
                  /*ThreadPool=*/nullptr,
                  /*BackendKernelSelectorConfig=*/nullptr);
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
    if (!mlasAvailable()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;

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
    return true;
}

size_t mlasFlashAttentionBufferBytesPerThread(int q_block_size,
                                              int kv_block_size,
                                              int v_head_size)
{
    if (q_block_size <= 0 || kv_block_size <= 0 || v_head_size <= 0) return 0;
    // flashattn.cpp lays out the per-thread scratch as:
    //   l[q_block_size] + m[q_block_size]
    //   + intermediate[q_block_size * kv_block_size]
    //   + temp_output[q_block_size * v_head_size]
    const size_t q  = static_cast<size_t>(q_block_size);
    const size_t kv = static_cast<size_t>(kv_block_size);
    const size_t vd = static_cast<size_t>(v_head_size);
    return (q * (2 + kv + vd)) * sizeof(float);
}

bool mlasFlashAttention(const float* query, const float* key, const float* value,
                        float* output,
                        int batch_size, int num_heads,
                        int q_seq_len, int kv_seq_len,
                        int qk_head_size, int v_head_size,
                        float scale,
                        int q_block_size, int kv_block_size,
                        void* scratch, int thread_count)
{
    if (!mlasAvailable()) return false;
    if (batch_size <= 0 || num_heads <= 0) return false;
    if (q_seq_len <= 0 || kv_seq_len <= 0) return false;
    if (qk_head_size <= 0 || v_head_size <= 0) return false;
    if (q_block_size <= 0 || kv_block_size <= 0) return false;
    if (thread_count <= 0 || scratch == nullptr) return false;
    if (query == nullptr || key == nullptr || value == nullptr || output == nullptr)
        return false;

    MlasFlashAttentionThreadedArgs args;
    args.batch_size            = batch_size;
    args.num_heads             = num_heads;
    args.q_sequence_length     = q_seq_len;
    args.kv_sequence_length    = kv_seq_len;
    args.qk_head_size          = qk_head_size;
    args.v_head_size           = v_head_size;
    args.q_block_size          = q_block_size;
    args.kv_block_size         = kv_block_size;
    args.scale                 = scale;
    args.thread_count          = thread_count;
    args.buffer                = static_cast<float*>(scratch);
    args.buffer_size_per_thread = mlasFlashAttentionBufferBytesPerThread(
                                      q_block_size, kv_block_size, v_head_size);
    args.query                 = query;
    args.key                   = key;
    args.value                 = value;
    args.output                = output;

    MlasFlashAttention(&args, /*ThreadPool=*/nullptr);
    return true;
}

}}  // cv::dnn

#endif  // HAVE_MLAS
