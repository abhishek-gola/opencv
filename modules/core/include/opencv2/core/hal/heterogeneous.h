// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_HAL_HETEROGENEOUS_H
#define OPENCV_CORE_HAL_HETEROGENEOUS_H

/**
 * @file heterogeneous.h
 * @brief Preview C ABI for asynchronous non-CPU HAL providers.
 *
 * This API deliberately does not extend the pointer-based CPU HAL. A device pointer, an OpenCL
 * cl_mem, a dma-buf file descriptor, and a host pointer are not interchangeable even when they
 * have the same numeric value or refer to the same physical storage. Providers expose typed
 * memory objects, queues, events, and programs. OpenCV's memory fabric owns placement, access,
 * synchronization, and fallback policy.
 *
 * The interface is a design prototype. There is no compatibility guarantee yet.
 */

#include "opencv2/core/cvdef.h"
#include "opencv2/core/llapi/llapi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define CV_HAL_HETERO_NOEXCEPT CV_NOEXCEPT
#else
#define CV_HAL_HETERO_NOEXCEPT
#endif

#define CV_HAL_HETERO_ABI_VERSION 0u
#define CV_HAL_HETERO_API_VERSION 0u
#define CV_HAL_HETERO_MAX_RANK 32u

/** Every extensible descriptor starts with this header. Unknown tail fields must be ignored. */
typedef struct CvHalHeteroStructHeader
{
    uint32_t struct_size;
    uint32_t struct_version;
} CvHalHeteroStructHeader;

typedef enum CvHalHeteroStatus
{
    CV_HAL_HETERO_OK = 0,
    /** The request was declined before side effects. OpenCV may try another provider. */
    CV_HAL_HETERO_NOT_SUPPORTED = 1,
    CV_HAL_HETERO_BUSY = 2,
    CV_HAL_HETERO_TIMEOUT = 3,
    CV_HAL_HETERO_INVALID_ARGUMENT = -1,
    CV_HAL_HETERO_OUT_OF_MEMORY = -2,
    CV_HAL_HETERO_DEVICE_LOST = -3,
    CV_HAL_HETERO_PERMISSION_DENIED = -4,
    CV_HAL_HETERO_INTERNAL_ERROR = -5
} CvHalHeteroStatus;

typedef struct CvHalHeteroDevice_t* CvHalHeteroDevice;
typedef struct CvHalHeteroQueue_t* CvHalHeteroQueue;
typedef struct CvHalHeteroMemory_t* CvHalHeteroMemory;
typedef struct CvHalHeteroEvent_t* CvHalHeteroEvent;
typedef struct CvHalHeteroProgram_t* CvHalHeteroProgram;

/** Stable provider/device identity. It is not a process pointer or an ordinal. */
typedef struct CvHalHeteroUuid
{
    uint8_t bytes[16];
} CvHalHeteroUuid;

typedef enum CvHalHeteroDeviceKind
{
    CV_HAL_HETERO_DEVICE_OTHER = 0,
    CV_HAL_HETERO_DEVICE_CPU = 1,
    CV_HAL_HETERO_DEVICE_GPU = 2,
    CV_HAL_HETERO_DEVICE_NPU = 3,
    CV_HAL_HETERO_DEVICE_DSP = 4,
    CV_HAL_HETERO_DEVICE_REMOTE = 5
} CvHalHeteroDeviceKind;

typedef enum CvHalHeteroDeviceCapability
{
    CV_HAL_HETERO_CAP_ASYNC_COPY = UINT64_C(1) << 0,
    CV_HAL_HETERO_CAP_ASYNC_EXECUTE = UINT64_C(1) << 1,
    CV_HAL_HETERO_CAP_TIMELINE_EVENTS = UINT64_C(1) << 2,
    CV_HAL_HETERO_CAP_EXTERNAL_MEMORY = UINT64_C(1) << 3,
    CV_HAL_HETERO_CAP_EXTERNAL_EVENTS = UINT64_C(1) << 4,
    CV_HAL_HETERO_CAP_PEER_COPY = UINT64_C(1) << 5,
    CV_HAL_HETERO_CAP_PROTECTED_MEMORY = UINT64_C(1) << 6,
    CV_HAL_HETERO_CAP_UNIFIED_ADDRESSING = UINT64_C(1) << 7
} CvHalHeteroDeviceCapability;

typedef struct CvHalHeteroDeviceProperties
{
    CvHalHeteroStructHeader header;
    CvHalHeteroUuid provider_uuid;
    CvHalHeteroUuid device_uuid;
    uint32_t kind;                 /**< CvHalHeteroDeviceKind */
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t numa_node;
    uint64_t capabilities;         /**< CvHalHeteroDeviceCapability bits */
    uint64_t max_allocation_size;
    uint64_t timestamp_frequency;
    const char* name;              /**< Provider-owned, valid while the device is retained. */
    uint64_t reserved[8];
} CvHalHeteroDeviceProperties;

typedef enum CvHalHeteroMemoryProperty
{
    CV_HAL_HETERO_MEMORY_CPU_VISIBLE = UINT64_C(1) << 0,
    CV_HAL_HETERO_MEMORY_DEVICE_LOCAL = UINT64_C(1) << 1,
    CV_HAL_HETERO_MEMORY_HOST_COHERENT = UINT64_C(1) << 2,
    CV_HAL_HETERO_MEMORY_HOST_CACHED = UINT64_C(1) << 3,
    CV_HAL_HETERO_MEMORY_MANAGED_MIGRATION = UINT64_C(1) << 4,
    CV_HAL_HETERO_MEMORY_EXPORTABLE = UINT64_C(1) << 5,
    CV_HAL_HETERO_MEMORY_IMPORTABLE = UINT64_C(1) << 6,
    CV_HAL_HETERO_MEMORY_PROTECTED = UINT64_C(1) << 7,
    CV_HAL_HETERO_MEMORY_SPARSE = UINT64_C(1) << 8,
    CV_HAL_HETERO_MEMORY_PINNED = UINT64_C(1) << 9,
    CV_HAL_HETERO_MEMORY_ATOMIC_COHERENT = UINT64_C(1) << 10
} CvHalHeteroMemoryProperty;

/**
 * A heap describes placement and coherency, not a programming API. Two devices may advertise the
 * same physical heap UUID. OpenCV may then import one allocation into both providers and keep one
 * logical backing, while still inserting ownership/cache transitions.
 */
typedef struct CvHalHeteroHeapProperties
{
    CvHalHeteroStructHeader header;
    CvHalHeteroUuid heap_uuid;
    uint32_t heap_index;
    uint32_t minimum_alignment;
    uint64_t properties;           /**< CvHalHeteroMemoryProperty bits */
    uint64_t capacity;
    uint64_t available;
    uint64_t preferred_page_size;
    uint64_t reserved[8];
} CvHalHeteroHeapProperties;

typedef enum CvHalHeteroAccessFlag
{
    CV_HAL_HETERO_ACCESS_READ = 1u << 0,
    CV_HAL_HETERO_ACCESS_WRITE = 1u << 1,
    /** The complete requested range will be overwritten; old bytes in that range need not migrate. */
    CV_HAL_HETERO_ACCESS_DISCARD = 1u << 2,
    /** Fail instead of migrating or creating a new replica. */
    CV_HAL_HETERO_ACCESS_NO_MIGRATE = 1u << 3,
    /** Acquisition must not block the calling thread. */
    CV_HAL_HETERO_ACCESS_NONBLOCKING = 1u << 4
} CvHalHeteroAccessFlag;

typedef struct CvHalHeteroRange
{
    uint64_t offset;
    uint64_t size;
} CvHalHeteroRange;

typedef struct CvHalHeteroAllocationDesc
{
    CvHalHeteroStructHeader header;
    uint32_t heap_index;
    uint32_t flags;
    uint64_t size;
    uint64_t alignment;
    uint32_t export_handle_type;   /**< Zero or one provider-advertised handle type. */
    uint32_t reserved0;
    uint64_t reserved[8];
} CvHalHeteroAllocationDesc;

typedef struct CvHalHeteroMemoryView
{
    CvHalHeteroStructHeader header;
    CvHalHeteroMemory memory;
    CvHalHeteroRange range;
    uint64_t reserved[4];
} CvHalHeteroMemoryView;

/** FourCC-like values are used so future handle kinds do not require an ABI-breaking enum. */
#define CV_HAL_HETERO_HANDLE_TAG(a, b, c, d) \
    ((uint32_t)(uint8_t)(a) | ((uint32_t)(uint8_t)(b) << 8) | \
     ((uint32_t)(uint8_t)(c) << 16) | ((uint32_t)(uint8_t)(d) << 24))

#define CV_HAL_HETERO_HANDLE_DMA_BUF_FD CV_HAL_HETERO_HANDLE_TAG('D', 'M', 'A', 'B')
#define CV_HAL_HETERO_HANDLE_OPAQUE_FD  CV_HAL_HETERO_HANDLE_TAG('O', 'F', 'D', '0')
#define CV_HAL_HETERO_HANDLE_WIN32_NT   CV_HAL_HETERO_HANDLE_TAG('W', 'N', 'T', '0')
#define CV_HAL_HETERO_HANDLE_D3D12      CV_HAL_HETERO_HANDLE_TAG('D', '3', '1', '2')
#define CV_HAL_HETERO_HANDLE_AHARDWAREBUFFER CV_HAL_HETERO_HANDLE_TAG('A', 'H', 'W', 'B')
#define CV_HAL_HETERO_HANDLE_IOSURFACE  CV_HAL_HETERO_HANDLE_TAG('I', 'O', 'S', 'F')
#define CV_HAL_HETERO_HANDLE_HOST_POINTER CV_HAL_HETERO_HANDLE_TAG('H', 'P', 'T', 'R')

typedef enum CvHalHeteroHandleOwnership
{
    /** The exporter retains ownership; the importer must retain/duplicate if it outlives the call. */
    CV_HAL_HETERO_HANDLE_BORROWED = 0,
    /** The receiver owns this exact handle after a successful import; failure leaves it with caller. */
    CV_HAL_HETERO_HANDLE_TRANSFERRED = 1,
    /** The handle is already an independent duplicate owned by the receiver. */
    CV_HAL_HETERO_HANDLE_DUPLICATED = 2
} CvHalHeteroHandleOwnership;

typedef union CvHalHeteroNativeHandleValue
{
    int64_t fd;
    uint64_t integer;
    void* pointer;
} CvHalHeteroNativeHandleValue;

typedef struct CvHalHeteroExternalHandle
{
    CvHalHeteroStructHeader header;
    uint32_t type;
    uint32_t ownership;            /**< CvHalHeteroHandleOwnership */
    CvHalHeteroNativeHandleValue value;
    uint64_t allocation_size;
    uint64_t offset;
    uint64_t flags;
    CvHalHeteroUuid physical_backing_uuid;
    uint64_t reserved[6];
} CvHalHeteroExternalHandle;

/**
 * Provider-verified properties of a memory object. The backing UUID from this query, rather than
 * an address or untrusted import descriptor, is used to decide whether two views alias bytes.
 */
typedef struct CvHalHeteroMemoryProperties
{
    CvHalHeteroStructHeader header;
    CvHalHeteroUuid physical_backing_uuid;
    uint32_t heap_index;
    uint32_t reserved0;
    uint64_t size;
    uint64_t backing_offset;
    uint64_t properties;           /**< CvHalHeteroMemoryProperty bits. */
    uint64_t reserved[8];
} CvHalHeteroMemoryProperties;

typedef struct CvHalHeteroTensorDesc
{
    CvHalHeteroStructHeader header;
    uint32_t cv_type;
    uint32_t rank;
    const uint64_t* shape;
    const int64_t* strides_bytes;
    uint32_t layout;
    uint32_t flags;
    /** Zero means distinct; matching nonzero groups describe permitted input/output aliasing. */
    uint32_t alias_group;
    uint32_t reserved0;
    uint64_t reserved[5];
} CvHalHeteroTensorDesc;

typedef struct CvHalHeteroProgramDesc
{
    CvHalHeteroStructHeader header;
    const char* format;            /**< For example "opencv.graph-ir" or "spirv". */
    uint32_t format_version;
    uint32_t flags;
    const void* data;
    uint64_t data_size;
    uint64_t cache_key_hi;
    uint64_t cache_key_lo;
    uint64_t reserved[6];
} CvHalHeteroProgramDesc;

typedef struct CvHalHeteroBinding
{
    CvHalHeteroStructHeader header;
    uint32_t slot;
    uint32_t access;               /**< CvHalHeteroAccessFlag bits */
    CvHalHeteroMemoryView memory;
    CvHalHeteroTensorDesc tensor;
    uint64_t reserved[4];
} CvHalHeteroBinding;

typedef struct CvHalHeteroSupportInfo
{
    CvHalHeteroStructHeader header;
    uint32_t supported;
    uint32_t reason_code;
    uint64_t required_workspace;
    uint64_t estimated_time_ns;
    uint64_t reserved[6];
} CvHalHeteroSupportInfo;

/**
 * Provider ABI v0. All calls are thread-safe unless a queue is involved; calls on one queue are
 * externally serialized by OpenCV. A successful asynchronous call returns an event, where NULL is
 * the canonical already-complete event. NOT_SUPPORTED is legal only before observable side effects.
 */
typedef struct OpenCV_HAL_Heterogeneous_API_v0
{
    OpenCV_API_Header api_header;

    CvHalHeteroStatus (CV_API_CALL *device_count)(uint32_t* count) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *device_get)(uint32_t index, CvHalHeteroDevice* device) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *device_retain)(CvHalHeteroDevice device) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *device_release)(CvHalHeteroDevice device) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *device_properties)(
        CvHalHeteroDevice device, CvHalHeteroDeviceProperties* properties) CV_HAL_HETERO_NOEXCEPT;

    CvHalHeteroStatus (CV_API_CALL *heap_count)(
        CvHalHeteroDevice device, uint32_t* count) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *heap_properties)(
        CvHalHeteroDevice device, uint32_t index, CvHalHeteroHeapProperties* properties) CV_HAL_HETERO_NOEXCEPT;
    /** Two-call enumeration: pass NULL to query the required count. */
    CvHalHeteroStatus (CV_API_CALL *memory_external_handle_types)(
        CvHalHeteroDevice device, uint32_t heap_index,
        uint32_t* handle_types, uint32_t* count) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *event_external_handle_types)(
        CvHalHeteroDevice device, uint32_t* handle_types,
        uint32_t* count) CV_HAL_HETERO_NOEXCEPT;

    CvHalHeteroStatus (CV_API_CALL *queue_create)(
        CvHalHeteroDevice device, uint32_t flags, CvHalHeteroQueue* queue) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *queue_release)(CvHalHeteroQueue queue) CV_HAL_HETERO_NOEXCEPT;

    CvHalHeteroStatus (CV_API_CALL *memory_allocate)(
        CvHalHeteroDevice device, const CvHalHeteroAllocationDesc* desc,
        CvHalHeteroMemory* memory) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *memory_import)(
        CvHalHeteroDevice device, const CvHalHeteroExternalHandle* external,
        const CvHalHeteroAllocationDesc* desc, CvHalHeteroMemory* memory) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *memory_export)(
        CvHalHeteroMemory memory, uint32_t handle_type,
        CvHalHeteroExternalHandle* external) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *memory_properties)(
        CvHalHeteroMemory memory, CvHalHeteroMemoryProperties* properties) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *memory_retain)(CvHalHeteroMemory memory) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *memory_release)(CvHalHeteroMemory memory) CV_HAL_HETERO_NOEXCEPT;

    CvHalHeteroStatus (CV_API_CALL *memory_map)(
        CvHalHeteroQueue queue, const CvHalHeteroMemoryView* view, uint32_t access,
        const CvHalHeteroEvent* waits, uint32_t wait_count, void** host_pointer,
        CvHalHeteroEvent* ready) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *memory_unmap)(
        CvHalHeteroQueue queue, const CvHalHeteroMemoryView* view, uint32_t access,
        void* host_pointer, const CvHalHeteroEvent* waits, uint32_t wait_count,
        CvHalHeteroEvent* done) CV_HAL_HETERO_NOEXCEPT;
    /** Both views belong to this provider; cross-provider copies are orchestrated via import/map. */
    CvHalHeteroStatus (CV_API_CALL *memory_copy)(
        CvHalHeteroQueue queue, const CvHalHeteroMemoryView* source,
        const CvHalHeteroMemoryView* destination, const CvHalHeteroEvent* waits,
        uint32_t wait_count, CvHalHeteroEvent* done) CV_HAL_HETERO_NOEXCEPT;

    void (CV_API_CALL *event_retain)(CvHalHeteroEvent event) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *event_release)(CvHalHeteroEvent event) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *event_wait)(
        CvHalHeteroEvent event, uint64_t timeout_ns) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *event_export)(
        CvHalHeteroEvent event, uint32_t handle_type,
        CvHalHeteroExternalHandle* external) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *event_import)(
        CvHalHeteroDevice device, const CvHalHeteroExternalHandle* external,
        CvHalHeteroEvent* event) CV_HAL_HETERO_NOEXCEPT;

    CvHalHeteroStatus (CV_API_CALL *program_query)(
        CvHalHeteroDevice device, const CvHalHeteroProgramDesc* program,
        const CvHalHeteroTensorDesc* inputs, uint32_t input_count,
        const CvHalHeteroTensorDesc* outputs, uint32_t output_count,
        CvHalHeteroSupportInfo* support) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *program_compile)(
        CvHalHeteroDevice device, const CvHalHeteroProgramDesc* desc,
        CvHalHeteroProgram* program) CV_HAL_HETERO_NOEXCEPT;
    void (CV_API_CALL *program_release)(CvHalHeteroProgram program) CV_HAL_HETERO_NOEXCEPT;
    CvHalHeteroStatus (CV_API_CALL *program_submit)(
        CvHalHeteroQueue queue, CvHalHeteroProgram program,
        const CvHalHeteroBinding* bindings, uint32_t binding_count,
        const CvHalHeteroEvent* waits, uint32_t wait_count,
        CvHalHeteroEvent* done) CV_HAL_HETERO_NOEXCEPT;

    const char* (CV_API_CALL *last_error_message)(void) CV_HAL_HETERO_NOEXCEPT;
    void* reserved[16];
} OpenCV_HAL_Heterogeneous_API_v0;

typedef OpenCV_HAL_Heterogeneous_API_v0 OpenCV_HAL_Heterogeneous_API;

typedef const OpenCV_HAL_Heterogeneous_API* (CV_API_CALL *FN_opencv_hal_heterogeneous_init_t)(
    uint32_t requested_abi_version, uint32_t requested_api_version, void* reserved);

#ifdef CV_HAL_HETERO_BUILD_PLUGIN
CV_PLUGIN_EXPORTS
const OpenCV_HAL_Heterogeneous_API* CV_API_CALL opencv_hal_heterogeneous_init_v0(
    uint32_t requested_abi_version, uint32_t requested_api_version,
    void* reserved) CV_HAL_HETERO_NOEXCEPT;
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#undef CV_HAL_HETERO_NOEXCEPT

#endif // OPENCV_CORE_HAL_HETEROGENEOUS_H
