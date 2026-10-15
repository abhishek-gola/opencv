// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SRC_HAL_HETEROGENEOUS_HPP
#define OPENCV_CORE_SRC_HAL_HETEROGENEOUS_HPP

#include "opencv2/core/hal/heterogeneous.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cv { namespace hal { namespace hetero {

struct CV_EXPORTS ByteRange
{
    uint64_t offset;
    uint64_t size;

    bool validFor(uint64_t allocationSize) const;
    bool covers(uint64_t allocationSize) const;
    bool overlaps(const ByteRange& other) const;
};

enum class TransferKind
{
    VisibilityOnly,
    DirectCopy,
    PeerCopy,
    ExternalAlias,
    DirectHostDma,
    PinnedHostStaging,
    PageableHostStaging,
    Unsupported
};

/** Facts discovered from both endpoint providers. No backend name is special-cased. */
struct TransferCapabilities
{
    bool samePhysicalStorage = false;
    bool systemCoherent = false;
    bool cacheMaintenanceRequired = false;
    bool sameDevice = false;
    bool directCopy = false;
    bool peerCopy = false;
    bool externalImport = false;
    bool sourceHostVisible = false;
    bool destinationHostVisible = false;
    bool directHostDma = false;
    bool pinnedHostStaging = false;
    bool pageableHostStaging = false;
    bool protectedMemory = false;
    /** The selected device-only route preserves the source allocation's protection domain. */
    bool protectedRoute = false;
};

struct TransferPlan
{
    TransferKind kind = TransferKind::Unsupported;
    uint32_t byteCopyCount = 0;
    bool requiresFence = false;
    bool requiresCacheMaintenance = false;
};

CV_EXPORTS TransferPlan planTransfer(const TransferCapabilities& capabilities);

enum class AccessMode
{
    Read,
    Write,
    ReadWrite
};

enum class AcquireStatus
{
    Accepted,
    Busy,
    InvalidRange,
    MissingSource,
    InvalidReplica
};

enum class ReleaseOutcome
{
    Success,
    /** Submission did not begin and the destination is unchanged. */
    Aborted,
    /** Execution or a materialization may have partially overwritten its destination. */
    Failed
};

struct AcquireRequest
{
    size_t replica = 0;
    ByteRange range = {0, 0};
    AccessMode mode = AccessMode::Read;
    bool discard = false;

    AcquireRequest() = default;
    AcquireRequest(size_t replica_, ByteRange range_, AccessMode mode_, bool discard_)
        : replica(replica_), range(range_), mode(mode_), discard(discard_)
    {
    }
};

struct AcquirePlan
{
    AcquireStatus status = AcquireStatus::InvalidReplica;
    uint64_t lease = 0;
    uint64_t requiredEpoch = 0;
    size_t sourceReplica = static_cast<size_t>(-1);
    bool materializationRequired = false;
    bool preserveContents = false;
};

/**
 * Logical buffer state used by the future memory-fabric coordinator.
 *
 * A replica is a provider view. Views with the same storageId alias the same physical bytes, which
 * is the zero-copy case. Epochs are committed only after a successful release, so failed kernels
 * never publish partially-written data. The prototype tracks epochs for the whole allocation and
 * conservatively serializes writes; ByteRange is retained in the contract for later interval-level
 * tracking and for exact backend cache/copy operations.
 */
class CV_EXPORTS BufferState
{
public:
    explicit BufferState(uint64_t allocationSize);

    size_t addReplica(uint64_t storageId, uint64_t initialEpoch = 0);
    AcquirePlan acquire(const AcquireRequest& request);
    bool release(uint64_t lease, ReleaseOutcome outcome);

    uint64_t committedEpoch() const;
    uint64_t replicaEpoch(size_t replica) const;
    size_t activeLeaseCount() const;

private:
    struct Replica
    {
        uint64_t storageId;
        uint64_t epoch;
    };

    struct Lease
    {
        uint64_t id;
        size_t replica;
        ByteRange range;
        AccessMode mode;
        uint64_t requiredEpoch;
        bool materializationRequired;
    };

    bool isWriter(AccessMode mode) const;
    size_t latestReplica() const;
    void setStorageEpoch(uint64_t storageId, uint64_t epoch);

    uint64_t allocationSize_;
    uint64_t committedEpoch_;
    uint64_t nextLease_;
    std::vector<Replica> replicas_;
    std::vector<Lease> leases_;
};

}}} // namespace cv::hal::hetero

#endif // OPENCV_CORE_SRC_HAL_HETEROGENEOUS_HPP
