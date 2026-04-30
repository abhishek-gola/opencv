# Vendored from microsoft/onnxruntime onnxruntime/core/mlas/.
# License: MIT (see LICENSE).
#
# Builds MLAS as a static library for x86_64 Linux/macOS so that the OpenCV
# DNN module can call MlasGemm (and, in future, more MLAS kernels) directly
# without depending on a full ONNX Runtime install. Currently only x86_64 is
# wired up; arm64/aarch64/power/etc. paths are left for a follow-up because
# the OpenCV-level CPU profile we're tuning runs on x86_64.
#
# The source list mirrors ORT's onnxruntime_mlas.cmake. To keep upstream
# diffs small, the directory layout (inc/, lib/, lib/x86_64/, lib/intrinsics/
# avx{,2,512}/) matches ORT exactly.

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$"))
  message(STATUS "MLAS: skipping — non-x86_64 host (${CMAKE_SYSTEM_PROCESSOR}).")
  set(HAVE_MLAS 0 PARENT_SCOPE)
  return()
endif()
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
  message(STATUS "MLAS: skipping — only Linux/macOS x86_64 wired up.")
  set(HAVE_MLAS 0 PARENT_SCOPE)
  return()
endif()

enable_language(ASM)

set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(MLAS_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# --- hardware-agnostic core (matches ORT's base list) ---
set(mlas_common_srcs
  ${MLAS_SRC_DIR}/platform.cpp
  # threading.cpp REPLACED by threading_opencv.cpp (cv::parallel_for_ backed)
  ${CMAKE_CURRENT_SOURCE_DIR}/threading_opencv.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/halfgemm.cpp
  ${MLAS_SRC_DIR}/qgemm.cpp
  ${MLAS_SRC_DIR}/qdwconv.cpp
  ${MLAS_SRC_DIR}/convolve.cpp
  ${MLAS_SRC_DIR}/convsym.cpp
  ${MLAS_SRC_DIR}/pooling.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/reorder.cpp
  ${MLAS_SRC_DIR}/snchwc.cpp
  ${MLAS_SRC_DIR}/activate.cpp
  ${MLAS_SRC_DIR}/logistic.cpp
  ${MLAS_SRC_DIR}/tanh.cpp
  ${MLAS_SRC_DIR}/eltwise.cpp
  ${MLAS_SRC_DIR}/erf.cpp
  ${MLAS_SRC_DIR}/silu.cpp
  ${MLAS_SRC_DIR}/gelu.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/dequantize.cpp
  ${MLAS_SRC_DIR}/quantize.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
  ${MLAS_SRC_DIR}/qladd.cpp
  ${MLAS_SRC_DIR}/qlmul.cpp
  ${MLAS_SRC_DIR}/qpostprocessor.cpp
  ${MLAS_SRC_DIR}/qlgavgpool.cpp
  ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
  ${MLAS_SRC_DIR}/qnbitgemm.cpp
  ${MLAS_SRC_DIR}/qlutgemm.cpp
  ${MLAS_SRC_DIR}/flashattn.cpp
  ${MLAS_SRC_DIR}/cast.cpp
  ${MLAS_SRC_DIR}/rotary_embedding.cpp
  ${MLAS_SRC_DIR}/saturation_check.cpp
  ${MLAS_SRC_DIR}/q4_dq.cpp
  ${MLAS_SRC_DIR}/q4gemm.cpp
)

# --- x86_64 grouped by ISA flag (matches ORT) ---
set(mlas_srcs_sse2
  ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
  ${MLAS_SRC_DIR}/x86_64/DgemmKernelSse2.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
  ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
  ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
  ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
)
if(NOT APPLE)
  list(APPEND mlas_srcs_sse2 ${MLAS_SRC_DIR}/x86_64/cvtfp16a.S)
endif()
set_source_files_properties(${mlas_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

set(mlas_srcs_avx
  ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
  ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx.S
  ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx.S
  ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx.S
  ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
)
set_source_files_properties(${mlas_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

set(mlas_srcs_avx2
  ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAvx2.S
  ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx2.S
  ${MLAS_SRC_DIR}/x86_64/QgemmU8U8KernelAvx2.S
  ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvxVnni.S
  ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx2.S
  ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S
  ${MLAS_SRC_DIR}/x86_64/DgemmKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/SconvKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/TransKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/LogisticKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/TanhKernelFma3.S
  ${MLAS_SRC_DIR}/x86_64/ErfKernelFma3.S
  ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
  ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
  ${MLAS_SRC_DIR}/intrinsics/avx2/saturation_check_avx2.cpp
  ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2.cpp
  ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.cpp
  ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
)
if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 13.1 AND NOT APPLE)
  list(APPEND mlas_srcs_avx2 ${MLAS_SRC_DIR}/x86_64/cvtfp16Avx.S)
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11")
  set_source_files_properties(${mlas_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
else()
  set_source_files_properties(${mlas_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
endif()

set(mlas_srcs_avx512f
  ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx512F.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
  ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx512F.S
  ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx512F.S
  ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx512F.S
  ${MLAS_SRC_DIR}/x86_64/TransKernelAvx512F.S
  ${MLAS_SRC_DIR}/intrinsics/avx512/gelu_avx512f.cpp
  ${MLAS_SRC_DIR}/intrinsics/avx512/silu_avx512f.cpp
  ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
  ${MLAS_SRC_DIR}/intrinsics/avx512/sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp
)
set_source_files_properties(${mlas_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

set(mlas_srcs_avx512core
  ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Core.S
  ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Vnni.S
  ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx512Core.S
  ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx512Core.S
  ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp
)
set_source_files_properties(${mlas_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl")

set(mlas_srcs_avx512vnni
  ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp
)
set_source_files_properties(${mlas_srcs_avx512vnni} PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl -mavx512f")

set(mlas_srcs_x86_64_misc
  ${MLAS_SRC_DIR}/dwconv.cpp
  ${MLAS_SRC_DIR}/dgemm.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
  ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
)
set_source_files_properties(${MLAS_SRC_DIR}/q4gemm_avx512.cpp
  PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl -mavx512f")

# AMX U8S8 GEMM kernel (Sapphire Rapids+). Required because platform.cpp
# unconditionally references MlasGemmU8S8DispatchAmx in the dispatch table;
# the kernel is itself only invoked at runtime if AMX is detected.
if(NOT APPLE)
  list(APPEND mlas_srcs_x86_64_misc
    ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
    ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S
  )
  set_source_files_properties(${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
    PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
  set_source_files_properties(${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S
    PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
endif()

set(MLAS_ALL_SOURCES
  ${mlas_common_srcs}
  ${mlas_srcs_sse2}
  ${mlas_srcs_avx}
  ${mlas_srcs_avx2}
  ${mlas_srcs_avx512f}
  ${mlas_srcs_avx512core}
  ${mlas_srcs_avx512vnni}
  ${mlas_srcs_x86_64_misc}
)

# OBJECT library — its objects will be linked directly into opencv_dnn so we
# don't need a separate .a / install rule.
add_library(opencv_dnn_mlas OBJECT ${MLAS_ALL_SOURCES})
set_target_properties(opencv_dnn_mlas PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(opencv_dnn_mlas PUBLIC
  ${MLAS_INC_DIR}
)
target_include_directories(opencv_dnn_mlas PRIVATE
  ${MLAS_SRC_DIR}
)
# Build MLAS standalone — without ORT's CPUIDInfo / threadpool / float16
# headers. Upstream gates those on this define (mlasi.h:162).
# MLAS_OPENCV_THREADING tells our patched mlasi.h::MlasGetMaximumThreadCount
# to return cv::getNumThreads() instead of the standalone fallback `1`,
# and pairs with threading_opencv.cpp which routes MlasTrySimpleParallel
# through cv::parallel_for_.
target_compile_definitions(opencv_dnn_mlas PRIVATE
  BUILD_MLAS_NO_ONNXRUNTIME=1
  MLAS_OPENCV_THREADING=1
)
# threading_opencv.cpp pulls in opencv_core headers.
target_include_directories(opencv_dnn_mlas PRIVATE
  ${CMAKE_SOURCE_DIR}/modules/core/include
)
# MLAS warns a lot under OpenCV's strict flags. Soften globally.
# Also force-include <cstring>: some upstream MLAS TUs (e.g. qlgavgpool.cpp)
# use memcpy/memset transitively via <immintrin.h> which is libc-version
# specific. Forcing the include makes the build hermetic across glibc versions.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(opencv_dnn_mlas PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-w>"
    "$<$<COMPILE_LANGUAGE:CXX>:-include>"
    "$<$<COMPILE_LANGUAGE:CXX>:cstring>"
  )
endif()
# platform.cpp's AMX init uses syscall(2) without including <unistd.h>.
set_source_files_properties(${MLAS_SRC_DIR}/platform.cpp
  PROPERTIES COMPILE_FLAGS "-include unistd.h")

set(HAVE_MLAS 1 PARENT_SCOPE)
set(MLAS_OBJECTS $<TARGET_OBJECTS:opencv_dnn_mlas> PARENT_SCOPE)
set(MLAS_INCLUDE_DIRS ${MLAS_INC_DIR} PARENT_SCOPE)
