// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "hal_heterogeneous.hpp"

#include <algorithm>
#include <limits>

namespace cv { namespace hal { namespace hetero {

bool ByteRange::validFor(uint64_t allocationSize) const
{
    return offset <= allocationSize && size <= allocationSize - offset;
}

bool ByteRange::covers(uint64_t allocationSize) const
{
    return offset == 0 && size == allocationSize;
}

bool ByteRange::overlaps(const ByteRange& other) const
{
    if (size == 0 || other.size == 0)
        return false;
    // Valid ranges are required at the API boundary, so subtraction avoids overflowing endpoints.
    return offset < other.offset ? other.offset - offset < size : offset - other.offset < other.size;
}

TransferPlan planTransfer(const TransferCapabilities& c)
{
    TransferPlan plan;
    if (c.protectedMemory && !c.protectedRoute)
        return plan;
    if (c.samePhysicalStorage)
    {
        plan.kind = TransferKind::VisibilityOnly;
        plan.requiresFence = true;
        plan.requiresCacheMaintenance = !c.systemCoherent || c.cacheMaintenanceRequired;
        return plan;
    }
    // Co-location is not a copy capability: a provider must positively advertise the route.
    if (c.directCopy)
    {
        plan.kind = TransferKind::DirectCopy;
        plan.byteCopyCount = 1;
        plan.requiresFence = true;
        return plan;
    }
    if (c.peerCopy)
    {
        plan.kind = TransferKind::PeerCopy;
        plan.byteCopyCount = 1;
        plan.requiresFence = true;
        return plan;
    }
    if (c.externalImport)
    {
        plan.kind = TransferKind::ExternalAlias;
        plan.requiresFence = true;
        plan.requiresCacheMaintenance = c.cacheMaintenanceRequired;
        return plan;
    }
    // Protected data cannot pass through a host-visible route, even when that route is pinned.
    if (c.protectedMemory)
        return plan;
    if ((c.sourceHostVisible || c.destinationHostVisible) && c.directHostDma)
    {
        plan.kind = TransferKind::DirectHostDma;
        plan.byteCopyCount = 1;
        plan.requiresFence = true;
        return plan;
    }
    if (c.pinnedHostStaging)
    {
        plan.kind = TransferKind::PinnedHostStaging;
        plan.byteCopyCount = 2;
        plan.requiresFence = true;
        return plan;
    }
    if (c.pageableHostStaging)
    {
        plan.kind = TransferKind::PageableHostStaging;
        plan.byteCopyCount = 2;
        plan.requiresFence = true;
        return plan;
    }
    return plan;
}

BufferState::BufferState(uint64_t allocationSize)
    : allocationSize_(allocationSize), committedEpoch_(0), nextLease_(1)
{
}

size_t BufferState::addReplica(uint64_t storageId, uint64_t initialEpoch)
{
    uint64_t sharedEpoch = initialEpoch;
    for (const Replica& replica : replicas_)
    {
        if (replica.storageId == storageId)
            sharedEpoch = std::max(sharedEpoch, replica.epoch);
    }
    committedEpoch_ = std::max(committedEpoch_, sharedEpoch);
    replicas_.push_back(Replica{storageId, sharedEpoch});
    setStorageEpoch(storageId, sharedEpoch);
    return replicas_.size() - 1;
}

AcquirePlan BufferState::acquire(const AcquireRequest& request)
{
    AcquirePlan plan;
    if (request.replica >= replicas_.size())
        return plan;
    if (!request.range.validFor(allocationSize_))
    {
        plan.status = AcquireStatus::InvalidRange;
        return plan;
    }

    Replica& target = replicas_[request.replica];
    const bool writer = isWriter(request.mode);
    const bool empty = request.range.size == 0;
    const bool fullDiscard = writer && request.discard && request.range.covers(allocationSize_);
    const bool needsPriorContents = !empty && (request.mode != AccessMode::Write || !fullDiscard);
    if (needsPriorContents && committedEpoch_ == 0)
    {
        plan.status = AcquireStatus::MissingSource;
        return plan;
    }

    const bool materializationRequired = needsPriorContents && target.epoch != committedEpoch_;
    const size_t sourceReplica = materializationRequired ? latestReplica() : request.replica;
    if (materializationRequired && sourceReplica == static_cast<size_t>(-1))
    {
        plan.status = AcquireStatus::MissingSource;
        return plan;
    }

    for (const Lease& active : leases_)
    {
        // A materialization is logically a read but writes its stale destination. Serialize it in
        // v0 just like a user write; ordinary reads of already-current storage may overlap.
        if (writer || materializationRequired || isWriter(active.mode) ||
            active.materializationRequired)
        {
            plan.status = AcquireStatus::Busy;
            return plan;
        }
    }

    if (nextLease_ == std::numeric_limits<uint64_t>::max())
        CV_Error(Error::StsOutOfRange, "heterogeneous HAL access lease ID overflow");
    plan.status = AcquireStatus::Accepted;
    plan.lease = nextLease_++;
    plan.requiredEpoch = committedEpoch_;
    plan.preserveContents = writer && !empty && !fullDiscard;
    plan.materializationRequired = materializationRequired;
    plan.sourceReplica = sourceReplica;

    leases_.push_back(Lease{plan.lease, request.replica, request.range, request.mode,
                            plan.requiredEpoch, plan.materializationRequired});
    return plan;
}

bool BufferState::release(uint64_t lease, ReleaseOutcome outcome)
{
    const auto it = std::find_if(leases_.begin(), leases_.end(),
                                 [lease](const Lease& value) { return value.id == lease; });
    if (it == leases_.end())
        return false;

    const Lease completed = *it;
    leases_.erase(it);
    if (outcome == ReleaseOutcome::Aborted)
        return true;

    Replica& target = replicas_[completed.replica];
    if (outcome == ReleaseOutcome::Failed)
    {
        // A kernel or copy may have modified only part of the physical backing. Every aliased view
        // is poisoned; a separate current replica may still be used to restore it later.
        if (isWriter(completed.mode) || completed.materializationRequired)
            setStorageEpoch(target.storageId, 0);
        return true;
    }

    // An empty operation has no logical side effect and therefore creates no new version.
    if (isWriter(completed.mode) && completed.range.size != 0)
    {
        if (committedEpoch_ == std::numeric_limits<uint64_t>::max())
            CV_Error(Error::StsOutOfRange, "heterogeneous HAL buffer epoch overflow");
        ++committedEpoch_;
        setStorageEpoch(target.storageId, committedEpoch_);
    }
    else if (completed.materializationRequired)
    {
        setStorageEpoch(target.storageId, completed.requiredEpoch);
    }
    return true;
}

uint64_t BufferState::committedEpoch() const
{
    return committedEpoch_;
}

uint64_t BufferState::replicaEpoch(size_t replica) const
{
    CV_Assert(replica < replicas_.size());
    return replicas_[replica].epoch;
}

size_t BufferState::activeLeaseCount() const
{
    return leases_.size();
}

bool BufferState::isWriter(AccessMode mode) const
{
    return mode == AccessMode::Write || mode == AccessMode::ReadWrite;
}

size_t BufferState::latestReplica() const
{
    for (size_t i = 0; i < replicas_.size(); ++i)
    {
        if (replicas_[i].epoch == committedEpoch_)
            return i;
    }
    return static_cast<size_t>(-1);
}

void BufferState::setStorageEpoch(uint64_t storageId, uint64_t epoch)
{
    for (Replica& replica : replicas_)
    {
        if (replica.storageId == storageId)
            replica.epoch = epoch;
    }
}

}}} // namespace cv::hal::hetero
