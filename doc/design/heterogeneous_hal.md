# Heterogeneous HAL v2 design spike

Status: exploratory, not an accepted OpenCV API

Baseline: OpenCV `5.x` at `e13b74e9a7862bdbed530415f8955e8b0d54b613` (2026-10-15)

Prototype branch: `design/heterogeneous-hal-v2`

## Decision summary

Add a new heterogeneous execution and memory layer alongside the existing CPU HAL. Do not extend
the pointer-based CPU HAL and do not dispatch by `UMat` allocator identity. The new layer consists
of:

1. a versioned C provider ABI for devices, memory heaps, queues, events, copies, external-memory
   interop, and compiled programs;
2. an OpenCV-owned memory fabric that tracks logical buffers, physical backings, provider views,
   committed epochs, access leases, and synchronization;
3. an operation planner that considers both execution and movement cost before selecting a GPU,
   NPU, DSP, CPU, or remote backend.

Zero-copy means that no bytes move. It never means that synchronization, cache maintenance,
ownership transfer, or lifetime management can be skipped.

## Why the current model does not scale

`UMatData` represents one host pointer and one untyped device `void*`, plus two stale-copy bits.
`MatAllocator` owns synchronous `map`, `unmap`, `upload`, `download`, and same-allocator `copy`
operations. This works for one host representation plus one device representation, but it cannot
describe:

- CPU + integrated GPU + NPU views of the same physical allocation;
- two discrete GPUs, including peer access or a staged route;
- a CUDA pointer versus `cl_mem`, Vulkan memory, dma-buf, D3D12 resource, IOSurface, or
  AHardwareBuffer without backend-specific casts;
- asynchronous completion and cross-queue dependencies;
- coherent, non-coherent, coarse-grain, fine-grain, or migratory shared memory;
- partial writes and multiple simultaneously cached device replicas;
- imported ownership, protected memory, device loss, or backend unload.

`UMat::getMat()` currently upgrades every request to read-write access, which can create needless
transfers and invalidations. `UMat::handle()` returns an allocator-dependent `void*`. T-API chooses
OpenCL from global/thread state and the presence of `UMat`, not from the actual memory type or an
operation/device capability query.

The current CUDA `MatAllocator` is also synchronous and performs one `cudaMemcpy` per continuous
plane. It is useful groundwork, but it is not a multi-device memory model.

## What PR #30259 contributes

[PR #30259](https://github.com/opencv/opencv/pull/30259) is open against 5.x and correctly exposes a
real safety failure: an OpenCL path can reinterpret another allocator's handle as `cl_mem`. Its
explicit cross-allocator copy path also avoids mapping a device destination merely to copy into it.
Those safety fixes should proceed independently.

The PR is intentionally narrower than this design. Its host-staging fallback still makes a
synchronous copy, and allocator equality is not a general capability test: a user allocator may be
OpenCL-compatible, and one allocator instance may serve several devices or heap types. The removed
fixed `HostDeviceBuffer` idea also cannot grow cleanly beyond a hard-coded host/CUDA/OpenCL list.

This design keeps the useful principles from the PR:

- never reinterpret an untyped handle based on the call site;
- validate backend compatibility before submission;
- make transfers explicit;
- test the actual OpenCL-enabled foreign-handle path.

It replaces allocator-name checks with typed handles and negotiated capabilities.

## Architecture

```text
OpenCV API / DNN / graph segment
              |
       operation adapter
              |
   placement + transfer planner
              |
  memory fabric and access leases
      /          |           \
 CPU provider  GPU provider  NPU provider ...
      \          |           /
    typed memory, queues, and events
```

The provider never decides global fallback. It reports support and costs, then executes a selected
plan. OpenCV never guesses what a provider handle means.

### Provider ABI

`opencv2/core/hal/heterogeneous.h` is a C ABI so an out-of-tree provider does not inherit OpenCV's
C++ ABI, STL, exception, or compiler settings. Every descriptor begins with `struct_size` and
`struct_version`; function tables have ABI/API versions and reserved tails.

The v0 surface has five opaque object types:

- device;
- queue;
- memory;
- event;
- compiled program.

Provider calls cover enumeration, heap properties, allocation, import/export, map/unmap, copy,
event import/export/wait, program query/compile/submit, and reference-counted lifetimes. A null event
is the canonical already-complete event. Other successful asynchronous operations return an event.
Supported external-memory and event handle types are enumerated rather than inferred from platform
or backend names. After allocation or import, `memory_properties` supplies the provider-verified
backing UUID, byte extent, heap, and offset; a UUID merely supplied in an import descriptor is not
proof of aliasing.

A provider copy consumes only its own memory handles. For a cross-provider route, the coordinator
first negotiates external-memory and external-event types, imports a typed view, or falls back to
provider-specific device/host legs through a staging allocation. It never hands provider A an
opaque memory or event object created by provider B.

`NOT_SUPPORTED` is a clean decline and is only legal before observable side effects. A device loss,
timeout, or execution failure after submission is an error, not a signal to replay on the CPU.
Blind replay could duplicate output or external side effects.

### Logical memory model

The memory fabric distinguishes four concepts:

- **logical buffer**: the value and byte extent visible to OpenCV;
- **physical backing**: storage that owns bytes;
- **provider view**: a typed way for one provider/device to address that backing;
- **replica**: a separate backing containing a version of the same logical value.

Two views with the same verified physical-backing UUID are aliases, not replicas. A host pointer and
an NPU I/O virtual address may therefore refer to one backing while remaining distinct typed views.
Equal numeric addresses, unified virtual addressing, or allocator pointers do not prove aliasing.
For an import, the coordinator accepts aliasing only when the export originated from its tracked
source object, the importer independently reports the same identity, and both providers agree on
the covered byte interval. Otherwise it treats the destination as a separate replica.

Each logical buffer has a monotonically increasing committed epoch. Each backing records the newest
epoch it contains. There can be several current backings after reads or copies. A successful write
creates a new epoch on its backing and makes other separate backings stale. Aliased views observe the
same backing epoch.

The prototype conservatively tracks an epoch for the whole allocation. A production optimization
may track disjoint dirty intervals, but it must not change the safety contract.

### Access leases

No backend receives memory until OpenCV grants a lease:

1. validate the byte range, tensor shape, strides, type, alignment, and bounds with overflow-safe
   arithmetic;
2. reject incompatible concurrent access;
3. select a current source backing if prior contents are required;
4. choose an alias, direct copy, peer copy, import, or staging route;
5. enqueue required waits, ownership transitions, cache operations, and copies;
6. hand the typed provider view to the operation only after its ready event;
7. release with the operation's completion event;
8. commit a new epoch only after successful completion;
9. distinguish an abort-before-submission from a failure that may have written bytes; the latter
   poisons its physical backing and every aliased view until a current replica restores it.

Concurrent reads of current backings are allowed. A read that must materialize a stale replica is
temporarily exclusive because its internal copy writes that destination. The v0 coordinator also
serializes any user write against all other leases and protects this state with its own lock. This
is deliberately conservative until interval-aware versioning and backend race guarantees exist.

`WRITE | DISCARD` avoids reading old data only when the lease covers the complete logical buffer.
A partial discard must preserve bytes outside its range, so a stale destination first materializes
the current version. This closes a common partial-update corruption hole.

### Integrated GPU/NPU and shared memory

The preferred zero-copy path is:

1. allocate from a heap explicitly advertised as visible to all required devices;
2. create or import typed views for CPU, GPU, and NPU;
3. verify that the views name the same physical backing;
4. pass readiness/completion events between queues;
5. perform cache maintenance or ownership transitions required by the heap;
6. keep the allocation alive until every outstanding event and imported view is released.

The heap capability is a vector, not one `SHARED` boolean:

- CPU-visible;
- device-local;
- host-coherent;
- host-cached;
- atomic-coherent;
- managed migration;
- pinned;
- importable/exportable;
- protected;
- sparse.

Important cases:

- **System coherent:** no byte copy and normally no explicit cache flush, but a queue/event ordering
  edge is still required between producer and consumer.
- **I/O coherent but not atomic coherent:** ordinary reads/writes are visible after fences, but
  cross-device atomics are rejected.
- **Non-coherent shared allocation:** no byte copy, but begin/end CPU access, flush/invalidate, and
  device ownership operations are mandatory.
- **Coarse-grain SVM:** mapping/unmapping is part of the access lease.
- **Fine-grain SVM:** explicit map may be unnecessary, but operation ordering and supported atomic
  scope still matter.
- **Managed/migratory memory:** one virtual address is not a performance guarantee. The planner may
  prefetch or choose an explicit replica when alternating CPU/device access would page-thrash.

The Linux dma-buf contract is a useful model: CPU cache-coherency bracketing and device execution
fences are separate responsibilities. `DMA_BUF_IOCTL_SYNC` does not by itself serialize a GPU.

### Discrete GPUs

CPU and discrete-GPU storage is physically separate unless an interconnect/import mechanism proves
otherwise. The goal is therefore the minimum necessary movement, not a false zero-copy promise.

Transfer route priority is:

1. same physical backing: zero-byte visibility transition;
2. same device/provider direct copy;
3. peer DMA between devices;
4. external-memory import of the same backing plus an external event/semaphore;
5. direct host-to-device or device-to-host DMA from registered/pinned host memory;
6. pipelined device-to-device transfer through a reusable pinned staging pool;
7. pageable host staging only when policy permits it;
8. unsupported if protected memory would need host staging.

Large staged transfers should be chunked and double-buffered so copy engines overlap. The cost model
must include setup cost, bandwidth, NUMA location, topology, cache operations, and whether the next
operation can consume the destination-resident replica. A one-off accelerator kernel should lose to
CPU execution when its transfer cost is larger than its compute gain.

### Operation dispatch

The provider interface accepts a versioned program/graph format rather than adding one permanent
function pointer for every OpenCV operation. Small operations can be encoded as one-node programs;
DNN or fused image-processing regions can be larger graphs.

Dispatch is split into:

- `program_query`: pure support and cost query, including types, layouts, shapes, strides,
  declared alias groups, determinism, workspace, and dynamic-shape limits;
- `program_compile`: side-effect-free with respect to user buffers and cacheable by content/device
  UUID;
- `program_submit`: explicit bindings, access modes, dependency events, and one completion event.

OpenCV validates every descriptor before the provider sees it. Unsupported dtype, layout, stride,
alias pattern, quantization, or dynamic shape is a normal query decline. A final catch-all type or
layout must never silently reinterpret newly-added OpenCV depths.

## Safety invariants

1. A native handle is consumed only by the provider that created or explicitly imported it.
2. Physical aliasing is asserted by an import/export contract and backing identity, never pointer
   equality.
3. A write is published only after its completion event succeeds.
4. Failed, cancelled, timed-out, or device-lost writes do not advance the committed epoch.
5. Host access and device execution are ordered separately from cache maintenance.
6. External-handle ownership is explicit: borrowed, transferred, or independently duplicated.
7. An allocation, plugin library, device, queue, and external owner outlive all dependent views and
   events.
8. A possibly-partial failed write poisons its physical backing; an abort known to precede all
   writes may preserve it.
9. Size, offset, shape, and stride arithmetic is checked before addition or multiplication.
10. Protected memory uses only routes proven to preserve its protection domain and never falls back
   through an ordinary host mapping.
11. User/custom allocators participate through capabilities and import, not an equality blacklist.
12. Fallback does not occur after a submission may have modified output.
13. The planner may reduce performance, but it may not weaken synchronization to claim zero-copy.

## Corner-case checklist

| Case | Required behavior |
|---|---|
| Empty/zero-byte tensor | Valid no-op; no dereference, transfer, or divide-by-zero. |
| Scalar/zero-rank tensor | Preserve OpenCV's scalar shape and nonzero element byte size. |
| Large dimensions | Check products, strides, offsets, and alignment in 64-bit before provider calls. |
| Non-contiguous tensor/ROI | Carry byte strides and base offset; decline if the provider cannot represent them. |
| Negative strides | Represent only when the program format and provider advertise support; otherwise materialize. |
| Overlapping source/destination | Declare in-place/overlap support during query; otherwise allocate a temporary. |
| Partial write to stale replica | Materialize current bytes unless the complete logical buffer is discarded. |
| Two readers on aliased CPU/NPU views | Allow after the producer fence/cache transition. |
| Reader plus writer | Serialize in v0, even for disjoint ranges. |
| Two devices from one provider | Device UUID and heap UUID remain part of every decision; allocator identity is insufficient. |
| P2P supported but disabled by topology/IOMMU | Re-query pair capability and choose external import or staging. |
| Mixed coherent and discrete GPUs | Maintain per-backing epochs; do not assign one global "unified" policy. |
| Managed memory oversubscription | Surface migration cost and permit prefetch/explicit replicas. |
| Cross-device atomics | Require advertised atomic coherence and scope; otherwise reject. |
| Backend disabled after allocation | Existing objects remain valid for release; new query/submit may decline. |
| Hot-unplug/device reset | Mark device lost, fail dependent events, retain enough state to release resources safely. |
| OOM during materialization | Leave the previous committed backing valid and return OOM before submission. |
| Kernel failure after write begins | Abort the epoch; output is poisoned/unavailable until overwritten, not CPU-replayed. |
| Cancellation | Cooperative; completion event defines when buffers may be reused. |
| Imported dma-buf/FD | Respect consumed-versus-borrowed ownership and retry interruptible CPU-sync ioctls. |
| Windows NT/KMT handle | Model differing ownership/reference rules explicitly. |
| External suballocation | Validate dedicated-allocation requirements, base offset, size, and alignment. |
| Plugin unload | Prohibited while any device, memory, event, queue, or program object remains. |
| Process shutdown | Avoid callbacks into unloaded runtimes; provide leak-safe termination behavior. |
| `fork()` | Child treats inherited provider state as invalid unless the provider explicitly supports it. |
| Protected/confidential buffer | No CPU map, debug dump, pageable staging, or untrusted provider. |
| Remote accelerator | Events may fail independently; network copies are explicit replicas, never aliases. |
| Endianness/device format | Program query must agree on element representation; byte identity alone is insufficient. |
| New OpenCV dtype/layout | Unknown values are rejected, not routed through a historical default branch. |
| Diagnostics | Include provider/device UUID, route, copied bytes, waits, and fallback reason without raw secrets. |

## Migration from `Mat`/`UMat`

No flag-day ABI break is required.

### Phase 0: design spike (this branch)

- Add the preview provider ABI.
- Add a pure transfer planner and logical-buffer state model.
- Test alias epochs, visibility-only plans, discrete staging, access conflicts, partial writes, failed
  writes, protected-memory rules, and overflow-safe ranges.

### Phase 1: provider registry and mock backend

- Load providers with OpenCV's plugin loader and pin the library for object lifetime.
- Add a CPU provider and deterministic mock GPU/NPU providers.
- Add event tracing, fault injection, and transfer accounting.

### Phase 2: `UMat` bridge

- Store a memory-fabric object in a sidecar initially reachable through allocator context, leaving
  the existing `UMatData` layout unchanged.
- Mirror a legacy host pointer/device handle only for old call sites.
- Add a typed internal acquire/release API; migrate `UMat::getMat()` and `handle()` callers.
- Preserve PR #30259's foreign-handle guards until no untyped OpenCL call remains.

### Phase 3: first operations

- Integrate copy, convert, resize, color conversion, and element-wise operations through support
  queries.
- Keep the existing CPU HAL as a provider implementation/fallback, not as a device-memory ABI.
- Add residency-aware performance tests before broad rollout.

### Phase 4: DNN and graph regions

- Bind DNN tensors to memory-fabric buffers.
- Compile supported graph regions and keep intermediates resident.
- Replace backend-local host/device wrappers with leases and events.

### Phase 5: public object model

- After provider and migration experience, decide whether `UMat` evolves or a new buffer/tensor type
  becomes public.
- Stabilize ABI only after Linux, Windows, macOS/iOS, Android, integrated NPU, and discrete
  multi-GPU implementations validate the same contract.

## Validation required before proposal

- Header compiles as both C and C++ on 32-bit and 64-bit targets.
- ThreadSanitizer tests for lease acquisition/release and provider teardown.
- Address/UndefinedBehaviorSanitizer tests for descriptor arithmetic.
- Mock fault injection at allocate, import, copy, submit, wait, release, and device-loss points.
- Integrated shared-memory test proving zero copied bytes plus correct cache/fence calls.
- Discrete GPU test proving one host/device DMA for pinned host memory and two transfers—not extra
  `Mat` copies—for GPU-to-GPU staging without P2P.
- Cross-API import/export tests for handle ownership and external events.
- ROI, non-contiguous, aliasing, in-place, empty, dynamic-shape, all-dtype, and partial-write tests.
- End-to-end benchmarks that report compute time, movement time, synchronization time, bytes moved,
  page faults, cache maintenance, warm/cold runs, and variance.

## Open design questions

- Which portable program IR should v0 require, if any, beyond a provider-named opaque format?
- Should interval-level dirty tracking be implemented before the first public release?
- Which external handle types are mandatory per platform?
- Should selection policy be process-wide, per graph, or per execution context?
- How should deterministic/numerically-equivalent policy be expressed across devices?
- Is the long-term public type an evolved `UMat`, a tensor/buffer object, or only an internal DNN and
  graph facility?

## Primary references

- [OpenCV PR #30259](https://github.com/opencv/opencv/pull/30259), explicit UMat transfers and
  foreign OpenCL-handle safeguards.
- [OpenCV PR #29807](https://github.com/opencv/opencv/pull/29807), DNN CUDA UMat consumer motivating
  the transfer work.
- [Linux dma-buf documentation](https://docs.kernel.org/driver-api/dma-buf.html), cache-coherency
  bracketing, implicit fences, and sync-file import/export.
- [OpenCL 3.0 API specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html),
  coarse/fine-grain SVM, map/unmap synchronization points, and SVM atomics.
- [CUDA unified memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html),
  hardware/software coherence, migration, and mixed coherent/discrete systems.
- [CUDA external resource interoperability](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html#external-resource-interoperability),
  external memory, mapping compatibility, semaphore synchronization, and OS-handle ownership.
- [Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html), external
  memory/semaphore transference and host/device synchronization requirements.
