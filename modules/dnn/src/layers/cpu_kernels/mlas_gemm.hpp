// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Thin wrapper around MlasGemm / MlasGemmBatch from the vendored MLAS
// library (modules/dnn/mlas/). Exists so the rest of the DNN module can ask
// for an SGEMM without including MLAS's internal headers (which pull in a
// lot of platform-detection #includes), and so we have a single chokepoint
// to tune the dispatch heuristic against fast_gemm.

#ifndef OPENCV_DNN_MLAS_GEMM_HPP
#define OPENCV_DNN_MLAS_GEMM_HPP

#include <cstddef>

namespace cv { namespace dnn {

#ifdef HAVE_MLAS

// True if the host can use MLAS at runtime. (Compile-time HAVE_MLAS only
// signals that we built the lib; a future build may target a CPU baseline
// MLAS rejects, in which case this will return false and callers fall back.)
bool mlasAvailable();

// Single SGEMM: C := alpha * op(A) * op(B) + beta * C
// where op(X) is X or X^T per trans_a / trans_b.
//
// A is interpreted as (trans_a ? K-by-M : M-by-K), row-major, leading dim
// `lda` (typically the storage row stride of the un-transposed view).
// B and C analogous.
//
// Returns true if MLAS executed the call, false to signal the caller to
// fall back (the wrapper currently always returns true on supported hosts;
// the bool is reserved for future thresholds where small GEMMs are better
// served by fast_gemm).
bool mlasSgemm(bool trans_a, bool trans_b,
               int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc);

// Batched SGEMM with per-batch offsets in elements (matching fastGemmBatch's
// convention). All M/N/K and strides are shared across the batch.
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
                    float* C_base, int ldc);

#else  // HAVE_MLAS

inline bool mlasAvailable() { return false; }

inline bool mlasSgemm(bool, bool, int, int, int, float,
                      const float*, int, const float*, int,
                      float, float*, int) { return false; }

inline bool mlasSgemmBatch(size_t, const size_t*, const size_t*, const size_t*,
                           bool, bool, int, int, int, float,
                           const float*, int, const float*, int,
                           float, float*, int) { return false; }

#endif  // HAVE_MLAS

}}  // cv::dnn

#endif  // OPENCV_DNN_MLAS_GEMM_HPP
