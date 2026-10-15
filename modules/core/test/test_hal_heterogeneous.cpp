// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "../src/hal_heterogeneous.hpp"

namespace opencv_test { namespace {

using namespace cv::hal::hetero;

TEST(Core_HeterogeneousHAL, transfer_planner_distinguishes_visibility_from_copy)
{
    TransferCapabilities shared;
    shared.samePhysicalStorage = true;
    shared.systemCoherent = true;
    TransferPlan plan = planTransfer(shared);
    EXPECT_EQ(TransferKind::VisibilityOnly, plan.kind);
    EXPECT_EQ(0u, plan.byteCopyCount);
    EXPECT_TRUE(plan.requiresFence);
    EXPECT_FALSE(plan.requiresCacheMaintenance);

    shared.systemCoherent = false;
    shared.cacheMaintenanceRequired = true;
    plan = planTransfer(shared);
    EXPECT_EQ(TransferKind::VisibilityOnly, plan.kind);
    EXPECT_EQ(0u, plan.byteCopyCount);
    EXPECT_TRUE(plan.requiresCacheMaintenance);

    TransferCapabilities hostToDiscrete;
    hostToDiscrete.sourceHostVisible = true;
    hostToDiscrete.directHostDma = true;
    plan = planTransfer(hostToDiscrete);
    EXPECT_EQ(TransferKind::DirectHostDma, plan.kind);
    EXPECT_EQ(1u, plan.byteCopyCount);

    TransferCapabilities discreteToDiscrete;
    discreteToDiscrete.pinnedHostStaging = true;
    plan = planTransfer(discreteToDiscrete);
    EXPECT_EQ(TransferKind::PinnedHostStaging, plan.kind);
    EXPECT_EQ(2u, plan.byteCopyCount);

    TransferCapabilities sameDeviceWithoutCopyCapability;
    sameDeviceWithoutCopyCapability.sameDevice = true;
    plan = planTransfer(sameDeviceWithoutCopyCapability);
    EXPECT_EQ(TransferKind::Unsupported, plan.kind);
}

TEST(Core_HeterogeneousHAL, transfer_planner_never_stages_protected_memory)
{
    TransferCapabilities capabilities;
    capabilities.protectedMemory = true;
    capabilities.directCopy = true;
    capabilities.pinnedHostStaging = true;
    capabilities.pageableHostStaging = true;

    TransferPlan plan = planTransfer(capabilities);
    EXPECT_EQ(TransferKind::Unsupported, plan.kind);
    EXPECT_EQ(0u, plan.byteCopyCount);

    capabilities.protectedRoute = true;
    plan = planTransfer(capabilities);
    EXPECT_EQ(TransferKind::DirectCopy, plan.kind);
}

TEST(Core_HeterogeneousHAL, aliased_views_share_one_committed_epoch)
{
    BufferState buffer(1024);
    const size_t cpu = buffer.addReplica(7, 1);
    const size_t npu = buffer.addReplica(7, 0);  // another view of the same physical allocation
    const size_t gpu = buffer.addReplica(9, 0);  // separate discrete storage

    EXPECT_EQ(1u, buffer.replicaEpoch(cpu));
    EXPECT_EQ(1u, buffer.replicaEpoch(npu));
    EXPECT_EQ(0u, buffer.replicaEpoch(gpu));

    AcquirePlan sharedRead = buffer.acquire({npu, {0, 1024}, AccessMode::Read, false});
    ASSERT_EQ(AcquireStatus::Accepted, sharedRead.status);
    EXPECT_FALSE(sharedRead.materializationRequired);
    EXPECT_TRUE(buffer.release(sharedRead.lease, ReleaseOutcome::Success));

    AcquirePlan discreteRead = buffer.acquire({gpu, {0, 1024}, AccessMode::Read, false});
    ASSERT_EQ(AcquireStatus::Accepted, discreteRead.status);
    EXPECT_TRUE(discreteRead.materializationRequired);
    EXPECT_TRUE(discreteRead.sourceReplica == cpu || discreteRead.sourceReplica == npu);
    EXPECT_TRUE(buffer.release(discreteRead.lease, ReleaseOutcome::Success));
    EXPECT_EQ(1u, buffer.replicaEpoch(gpu));

    AcquirePlan gpuWrite = buffer.acquire({gpu, {0, 1024}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, gpuWrite.status);
    EXPECT_FALSE(gpuWrite.preserveContents);
    EXPECT_TRUE(buffer.release(gpuWrite.lease, ReleaseOutcome::Success));

    EXPECT_EQ(2u, buffer.committedEpoch());
    EXPECT_EQ(2u, buffer.replicaEpoch(gpu));
    EXPECT_EQ(1u, buffer.replicaEpoch(cpu));
    EXPECT_EQ(1u, buffer.replicaEpoch(npu));
}

TEST(Core_HeterogeneousHAL, failed_write_poisons_its_physical_backing)
{
    BufferState buffer(256);
    const size_t cpu = buffer.addReplica(1, 1);

    AcquirePlan write = buffer.acquire({cpu, {0, 256}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, write.status);
    ASSERT_TRUE(buffer.release(write.lease, ReleaseOutcome::Failed));
    EXPECT_EQ(1u, buffer.committedEpoch());
    EXPECT_EQ(0u, buffer.replicaEpoch(cpu));

    const AcquirePlan read = buffer.acquire({cpu, {0, 256}, AccessMode::Read, false});
    EXPECT_EQ(AcquireStatus::MissingSource, read.status);
}

TEST(Core_HeterogeneousHAL, aborted_write_preserves_unchanged_backing)
{
    BufferState buffer(256);
    const size_t cpu = buffer.addReplica(1, 1);

    AcquirePlan write = buffer.acquire({cpu, {0, 256}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, write.status);
    ASSERT_TRUE(buffer.release(write.lease, ReleaseOutcome::Aborted));
    EXPECT_EQ(1u, buffer.committedEpoch());
    EXPECT_EQ(1u, buffer.replicaEpoch(cpu));
}

TEST(Core_HeterogeneousHAL, partial_discard_preserves_unwritten_bytes)
{
    BufferState buffer(256);
    const size_t cpu = buffer.addReplica(1, 1);
    const size_t gpu = buffer.addReplica(2, 1);

    AcquirePlan gpuWrite = buffer.acquire({gpu, {0, 256}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, gpuWrite.status);
    ASSERT_TRUE(buffer.release(gpuWrite.lease, ReleaseOutcome::Success));

    AcquirePlan partial = buffer.acquire({cpu, {32, 64}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, partial.status);
    EXPECT_TRUE(partial.preserveContents);
    EXPECT_TRUE(partial.materializationRequired);
    ASSERT_TRUE(buffer.release(partial.lease, ReleaseOutcome::Aborted));

    AcquirePlan whole = buffer.acquire({cpu, {0, 256}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, whole.status);
    EXPECT_FALSE(whole.preserveContents);
    EXPECT_FALSE(whole.materializationRequired);
    ASSERT_TRUE(buffer.release(whole.lease, ReleaseOutcome::Success));
}

TEST(Core_HeterogeneousHAL, access_leases_serialize_writes)
{
    BufferState buffer(128);
    const size_t cpu = buffer.addReplica(1, 1);
    const size_t npu = buffer.addReplica(1, 0);

    AcquirePlan read1 = buffer.acquire({cpu, {0, 64}, AccessMode::Read, false});
    AcquirePlan read2 = buffer.acquire({npu, {64, 64}, AccessMode::Read, false});
    ASSERT_EQ(AcquireStatus::Accepted, read1.status);
    ASSERT_EQ(AcquireStatus::Accepted, read2.status);
    EXPECT_EQ(2u, buffer.activeLeaseCount());

    const AcquirePlan write = buffer.acquire({npu, {0, 128}, AccessMode::Write, true});
    EXPECT_EQ(AcquireStatus::Busy, write.status);
    EXPECT_TRUE(buffer.release(read1.lease, ReleaseOutcome::Success));
    EXPECT_TRUE(buffer.release(read2.lease, ReleaseOutcome::Success));
}

TEST(Core_HeterogeneousHAL, materializing_reads_are_exclusive)
{
    BufferState buffer(128);
    const size_t cpu = buffer.addReplica(1, 1);
    const size_t gpu = buffer.addReplica(2, 0);

    const AcquirePlan currentRead = buffer.acquire({cpu, {0, 128}, AccessMode::Read, false});
    ASSERT_EQ(AcquireStatus::Accepted, currentRead.status);
    const AcquirePlan materializingRead = buffer.acquire({gpu, {0, 128}, AccessMode::Read, false});
    EXPECT_EQ(AcquireStatus::Busy, materializingRead.status);
    EXPECT_TRUE(buffer.release(currentRead.lease, ReleaseOutcome::Success));
}

TEST(Core_HeterogeneousHAL, range_validation_is_overflow_safe)
{
    const ByteRange valid{96, 32};
    const ByteRange invalid{UINT64_MAX - 8, 32};
    EXPECT_TRUE(valid.validFor(128));
    EXPECT_FALSE(invalid.validFor(128));
    EXPECT_TRUE(valid.overlaps(ByteRange{100, 1}));
    EXPECT_FALSE(valid.overlaps(ByteRange{0, 96}));

    BufferState buffer(128);
    const size_t cpu = buffer.addReplica(1, 1);
    const AcquirePlan plan = buffer.acquire({cpu, invalid, AccessMode::Read, false});
    EXPECT_EQ(AcquireStatus::InvalidRange, plan.status);
}

TEST(Core_HeterogeneousHAL, empty_ranges_are_no_op_without_a_source)
{
    BufferState buffer(0);
    const size_t cpu = buffer.addReplica(1, 0);

    AcquirePlan read = buffer.acquire({cpu, {0, 0}, AccessMode::Read, false});
    ASSERT_EQ(AcquireStatus::Accepted, read.status);
    EXPECT_FALSE(read.materializationRequired);
    ASSERT_TRUE(buffer.release(read.lease, ReleaseOutcome::Success));

    AcquirePlan write = buffer.acquire({cpu, {0, 0}, AccessMode::Write, true});
    ASSERT_EQ(AcquireStatus::Accepted, write.status);
    ASSERT_TRUE(buffer.release(write.lease, ReleaseOutcome::Success));
    EXPECT_EQ(0u, buffer.committedEpoch());
}

}} // namespace opencv_test
