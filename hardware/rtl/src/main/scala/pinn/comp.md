# RTL Wrapper Comparison Note

This note compares the current in-repo RTL structure for:

- `classical`
- `fusemax`
- `onesa`
- `pinnacle`
- `systolicattention`

It is about the current repo implementation, not the original papers.

The main question is no longer "which designs even have a scratchpad wrapper?" The current repo now exposes a fairly aligned structural datapath for all five designs at the `scratchpad + array + accumulator-side block` level, and the top-level scratchpads now converge on the same common `Scratchpad` abstraction directly or indirectly. The real mismatch is:

- PE granularity
- accumulator compute style
- controller placement
- scratchpad arbitration / interface semantics
- whether the wrapper is a full system or only a structurally exposed datapath

Short answer:

- The five designs are comparable at compute-core scope.
- They are also comparable at accumulator-compute-unit scope under a fixed total memory budget.
- They are **not** all comparable as full systems, because only some include an integrated controller / sequencer.
- Wrapper-level module names alone are still not enough to claim apples-to-apples fairness.

## Scope of comparison

This note uses these current structural baselines:

- `classical`: `ClassicalSystem`
- `onesa`: `OneSaSystem`
- `fusemax`: `FuseMaxSystem`
- `pinnacle`: `PinnacleSystem`
- `systolicattention`: `SystolicAttentionSystem`

Important exception:

- `SystolicAttentionSystem` is now the public top-level. The lower-level externally controlled block is
  `SystolicAttentionDatapath`.

Working comparison assumption for the paper:

- the total memory space of `scratchpad + accumulator` is treated as identical across implementations
- memory capacity itself is therefore **not** the main differentiator in this note
- the main focus is the compute unit, especially the compute capability placed in or around the accumulator

Important consequence:

- some wrappers are full tile systems
- some designs flatten functionality that other designs keep in separate modules

So "has block X somewhere" and "has a directly equivalent module boundary for X" are still different statements.

## High-level summary

| Design              | Structural wrapper used here | Explicit SA block                   | Explicit scratchpad in wrapper | Explicit accumulator-side block                  | Integrated controller / sequencer | Fairly comparable as full system? |
| ------------------- | ---------------------------- | ----------------------------------- | ------------------------------ | ------------------------------------------------ | --------------------------------- | --------------------------------- |
| `classical`         | `ClassicalSystem`            | Yes                                 | Yes                            | Yes                                              | Yes                               | Yes                               |
| `onesa`             | `OneSaSystem`                | Yes                                 | Yes                            | Yes                                              | No, only external control bundle  | No                                |
| `fusemax`           | `FuseMaxSystem`              | Yes, `FuseMax2DArray`               | Yes                            | Yes, `FuseMax1DArray`                            | No single integrated controller   | No                                |
| `pinnacle`          | `PinnacleSystem`             | Yes                                 | Yes                            | Yes, but much richer than classical              | Yes                               | Yes                               |
| `systolicattention` | `SystolicAttentionSystem`    | Yes, via inlined delayers + core SA | Yes                            | Yes, via `core.Accumulator` with integrated SRAM | No, only external control bundles | No                                |

## Per-design notes

### Classical

Relevant files:

- [`classical/wrapper.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/classical/wrapper.scala)
- [`classical/sa.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/classical/sa.scala)

Structure:

- `ClassicalSystem` instantiates:
  - `Scratchpad`
  - `SystolicArray`
  - `Accumulator`
- `ClassicalGemmController`

Assessment:

- This is still the cleanest reference baseline for a conventional SA decomposition.
- It is suitable as the reference point for:
  - classical SA compute core
  - scratchpad organization
  - ping-pong accumulator
  - explicit top-level controller

### ONE-SA

Relevant files:

- [`onesa/wrapper.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/onesa/wrapper.scala)
- [`onesa/sa.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/onesa/sa.scala)
- [`onesa/design.md`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/onesa/design.md)

Structure:

- The wrapper instantiates:
  - `Scratchpad`
  - `OneSaL3Rearrange`
  - `SystolicArray`
  - `Accumulator`
- `OneSaL3Rearrange` now owns the resident `k` / `b` table state and write ports.
- Control is provided through `OneSaSystemCtrl`; there is no integrated instruction sequencer or higher-level controller in the wrapper.

Assessment:

- The compute core is a modified classical array with:
  - SIMD payloads
  - diagonal computation PEs
  - nonlinear routing mode
- Each PE is not a scalar MAC PE anymore. In the current wrapper it uses `simdWidth = 2`, so one PE performs a 2-lane SIMD dot product in linear mode.
- The accumulator is still very close to `classical`.
- Compared with the older repo state, ONE-SA is now structurally closer to `classical` at datapath-wrapper scope because it also exposes an explicit scratchpad.

Fair comparison:

- Fair against `classical` for:
  - scratchpad + SA + minimal-accumulator datapath structure
  - systolic-array microarchitecture discussion
  - accumulator structure
- But the PE granularity is different from scalar-MAC arrays such as `classical` and most of `pinnacle`.
- So ONE-SA should not be compared by raw PE count alone. It should be normalized by:
  - SIMD lane count per PE
  - effective MACs per cycle
  - or total array throughput at matched data type / array size
- Not fair against `classical` or `pinnacle` as a full system, because the controller remains externalized.

### FuseMax

Relevant files:

- [`fusemax/wrapper.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/fusemax/wrapper.scala)
- [`fusemax/sa.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/fusemax/sa.scala)
- [`fusemax/design.md`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/fusemax/design.md)

Structure:

- The datapath wrapper instantiates:
  - `Scratchpad`
  - `FuseMax2DArray`, which corresponds to the systolic array
  - `FuseMax1DArray`, which corresponds to the accumulator-side compute unit
- The reusable `sa.scala` also contains a generic-looking `SystolicArray` and `Accumulator`, but the wrapper does not use those names directly for the full-array path.
- Control is command-driven and externally supplied through `FuseMaxSystemCtrl`; there is no single clean top-level controller module for the datapath wrapper.
- Some row reduction behavior is flattened into the wrapper.

Important difference from classical:

- The PE is command-driven, not just `LOAD_WEIGHT` / `COMPUTE`.
- Each PE has a 10-entry local RF.
- The 2D array drains row-local results into a 1D array.
- The 1D array is the accumulator-side compute unit for FuseMax. It acts like a running-softmax state engine rather than a classical ping-pong SRAM accumulator.

Assessment:

- FuseMax definitely has comparable systolic-array and accumulator-side compute components.
- For this note, `FuseMax2DArray` should be treated as the SA and `FuseMax1DArray` should be treated as the accumulator compute unit.
- The wrapper boundary is now clearer than before because it explicitly exposes:
  - scratchpad
  - 2D array
  - 1D accumulator-side block
- But the controller boundary is still external, and some reduction behavior is still flattened into the wrapper rather than isolated in a separate module.

Fair comparison:

- Fair for compute-core comparison if we compare:
  - `classical.SystolicArray`
  - `onesa.SystolicArray`
  - `fusemax.FuseMax2DArray`
- Fair for accumulator-side compute comparison if we compare `FuseMax1DArray` against the other designs' accumulator compute units.
- The caution is not that FuseMax lacks an accumulator. The caution is that its accumulator compute style is RF-centric and controller-driven rather than SRAM ping-pong based.
- Not fair to compare wrapper area or module count directly against designs that keep reductions or control in separate modules.

### PINNacle

Relevant files:

- [`pinnacle/wrapper.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/pinnacle/wrapper.scala)
- [`pinnacle/sa.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/pinnacle/sa.scala)
- [`pinnacle/design.md`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/pinnacle/design.md)

Structure:

- The wrapper instantiates:
  - `Scratchpad`
  - `PinnacleAccumulator`
  - `SystolicArray`
- The wrapper also contains the main controller / sequencer as an internal state machine.
- This is a real full tile system, not just a datapath shell.

Important difference from classical:

- The SA is modified with strip rows for the PINN approximator.
- The accumulator is much richer than the classical ping-pong buffer:
  - multiple SRAM bank groups
  - dedicated vector registers `M0`, `M1`, `alpha`, `L`, `RowSum`
  - affine / scale / update logic
  - attention-specific state handling

Assessment:

- PINNacle is structurally comparable to `classical` as a full system because both expose:
  - scratchpad
  - SA
  - accumulator
  - top-level controller
- But the accumulator is not the same class of module anymore. It is much more capable.

Fair comparison:

- Fair for full-system comparison against `classical`, if the paper clearly states that PINNacle adds functionality in ACC and controller.
- Fair for compute-core comparison against other modified arrays.
- Not fair to claim the accumulator is directly comparable to `classical` or `onesa` without noting the extra vector/state logic.

### SystolicAttention

Relevant files:

- [`systolicattention/wrapper.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/systolicattention/wrapper.scala)
- [`systolicattention/core.scala`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/systolicattention/core.scala)
- [`systolicattention/design.md`](/data/user/cxu930/projects/pinn-fullstack/rtl/src/main/scala/pinn/systolicattention/design.md)

The public top-level in this subsystem is `SystolicAttentionSystem`.

For structural comparison below, the lower-level block of interest is `SystolicAttentionDatapath`.

Structure of `SystolicAttentionDatapath`:

- `Scratchpad`
- inlined array-side datapath:
  - `InputDelayer`
  - `SystolicArray`
  - `OutputDelayer`
- `Accumulator`
  - internal `BankedSram`
- external control bundles:
  - `spCtrl`
  - `saCtrl`
  - `accCtrl`
  - `accMemCtrl`

Important difference from classical:

- the 2D array already includes:
  - bidirectional vertical dataflow
  - comparator row
  - in-PE exp2 support
- the accumulator-side block is not a classical tile accumulator
  - it is a memory-backed vector compute path with scale, exp, reciprocal, and fused accumulation modes driven by external control

Assessment:

- Compared with the older repo state, the structural datapath wrapper now clearly exposes:
  - scratchpad
  - array-side block, now flattened into the system rather than kept behind a local wrapper
  - accumulator-side block
- The scratchpad boundary is also now more aligned with `classical` / `onesa` / `fusemax`, although the shared host-read arbitration semantics still differ from design to design.
- That makes it more comparable than before at structural datapath scope.
- But the control boundary is still externalized, and the internal compute style is still much more specialized than `classical` or `onesa`.
- The functional `SystolicAttentionModel` should still not be used for structural component comparison against the others.

Fair comparison:

- Fair for compute-core comparison if using the in-system array-side datapath or the core SA itself, depending on whether input/output alignment logic is included in scope.
- Fair for accumulator-side compute comparison if the paper makes clear that this is a specialized online-softmax-style compute path rather than a minimal SRAM accumulator.
- Fair for scratchpad-interface comparison only if the paper also states that the controller and memory scheduling are still externalized.
- Not fair for full-system wrapper comparison because there is no integrated controller in the structural datapath wrapper.

## What is actually comparable

### 1. Systolic-array / compute-core comparison

This is still the fairest cross-design comparison.

Suggested mapping:

| Design              | Compute-core module to compare                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `classical`         | `classical.SystolicArray`                                                                                 |
| `onesa`             | `onesa.SystolicArray`                                                                                     |
| `fusemax`           | `fusemax.FuseMax2DArray` or PE substrate in `fusemax/sa.scala`                                            |
| `pinnacle`          | `pinnacle.SystolicArray`                                                                                  |
| `systolicattention` | `systolicattention.SystolicAttentionSystem` array-side datapath or `systolicattention.core.SystolicArray` |

This comparison is fair if the paper explicitly discusses:

- PE state
- extra datapaths
- extra local storage
- added nonlinear / softmax logic
- whether comparator rows or strip rows are inside the array

### 2. Scratchpad / memory-budget comparison

Under the working assumption of this note, the total memory space of `scratchpad + accumulator` is fixed across designs.

Implication:

- scratchpad capacity itself is not the main comparison target
- whether a wrapper exposes an explicit scratchpad module still matters for interface and integration discussion
- but scratchpad presence is **no longer** the main structural differentiator among the current wrappers used in this note

Useful structural observation:

- `classical`, `onesa`, `fusemax`, and `systolicattention` now instantiate the common `Scratchpad` helper directly in the structural wrapper path
- `pinnacle` now also instantiates the same common `Scratchpad` helper directly
- the real differences are:
  - which datapath read channels exist
  - whether the host read shares a datapath read port
  - read/write arbitration policy
  - controller placement
  - whether extra table or vector state sits beside the main scratchpad

So in the paper:

- keep memory capacity normalized
- compare memory placement / interface as an integration choice
- do not treat scratchpad presence alone as a fairness separator at current wrapper scope

### 3. Accumulator comparison

This note focuses on the **compute unit in the accumulator**, assuming total `scratchpad + accumulator` memory capacity is matched.

Under that assumption, all five designs have a meaningful accumulator-side comparison point, but their compute capability is very different.

#### Minimal accumulator compute

- `classical.Accumulator`
- `onesa.Accumulator`

These are directly comparable and structurally very close.

Characteristics:

- SRAM-backed ping-pong buffering
- `WRITE` / `ACCUM`
- minimal arithmetic around the memory

#### Specialized accumulator compute units

- `fusemax.FuseMax1DArray`
- `pinnacle.PinnacleAccumulator`
- `systolicattention.core.Accumulator`

These are all valid accumulator-side compute units, but they are not the same microarchitecture as `classical` / `onesa`.

Useful interpretation:

- `fusemax`: 1D PE-array recurrence engine with RF-based state and final division
- `pinnacle`: SRAM-backed accumulator plus vector-state registers and fused affine / reduction / scale logic
- `systolicattention`: vector accumulator with scale, exp, reciprocal, and fused accumulation modes, with dedicated SRAM integrated inside the block

So the fair paper comparison is:

- compare memory capacity under a fixed budget
- compare accumulator **compute capability** separately from raw memory size
- do not collapse these designs into one "same accumulator" category

### 4. Controller comparison

Directly comparable integrated controllers / sequencers:

- `classical.ClassicalSystem` internal `ClassicalGemmController`
- `pinnacle.PinnacleSystem` internal wrapper state machine

Partially comparable or not directly comparable:

- `fusemax`: mixed model, externally driven commands plus local control
- `onesa`: no integrated controller in the wrapper, only control / status bundles
- `systolicattention`: no integrated controller in the lower-level structural datapath wrapper

Important note:

- `SystolicAttentionSystem` owns the functional sequencing now, but the structural comparison target for this note is
  still the lower-level `SystolicAttentionDatapath`.

## Paper-writing guidance

Recommended wording:

- Do **not** say all five wrappers are partitioned identically.
- Do **not** say their scratchpad, accumulator, and controller blocks are directly equivalent.
- Do say the five designs can be compared at the compute-core level, and also at the accumulator-compute-unit level under a fixed total memory budget, but not all at the full-system level.
- Do separate comparisons into:
  - compute core
  - accumulator-side compute
  - controller / system integration

Safe claim:

- All five structural baselines in this note expose scratchpad + array-side + accumulator-side blocks.
- All five top-level scratchpads now reduce to the same common `Scratchpad` implementation path, but their exposed interfaces and arbitration semantics are still not identical.
- `classical` and `pinnacle` are the most comparable as full in-repo tile systems.
- `classical` and `onesa` are the most comparable at scratchpad + SA + minimal-accumulator datapath granularity.
- `fusemax` should be interpreted as `2D array = SA` and `1D array = accumulator`.
- `onesa` needs PE-granularity normalization because one PE contains SIMD MAC work.
- `systolicattention` is structurally more aligned than before at datapath-wrapper scope, but it is still externally controlled and internally much more specialized.

Unsafe claim:

- "All five designs have equivalent systolic array, scratchpad, accumulator, and controller modules and can be fairly compared component-by-component."

That statement is still not supported by the current RTL organization.

## Suggested comparison axes for the paper

Use these axes instead of raw wrapper-module names:

1. PE local state
2. PE granularity
3. Extra routing directions
4. Nonlinear / softmax logic placement
5. Need for comparator row or strip rows
6. On-chip memory organization
7. Accumulator compute style: SRAM-backed, RF-backed, or vector-state assisted
8. Controller placement
9. Whether the RTL is a full tile system or only a structurally exposed datapath

## Concrete TODOs

- Build a paper table with one row per design and these columns:
  - `PE state`
  - `PE granularity`
  - `extra datapath`
  - `scratchpad interface`
  - `accumulator compute unit`
  - `accumulator type`
  - `controller placement`
  - `full system or datapath only`
- For `onesa`, normalize comparisons by SIMD width or effective MAC throughput rather than raw PE count.
- For `systolicattention`, explicitly choose the structural baseline:
  - use `SystolicAttentionSystem` and the SA / accumulator blocks under it
  - do not use `SystolicAttentionModel` for structural component comparison
- For `fusemax`, state explicitly in the paper that:
  - `FuseMax2DArray` is the systolic array
  - `FuseMax1DArray` is the accumulator compute unit
- In the paper, separate accumulator memory capacity from accumulator compute capability.
- If area comparison is needed, normalize at the same scope:
  - PE only
  - full compute core only
  - accumulator compute unit only
  - full tile system only
  - never mix these scopes in one chart
- If controller complexity is discussed, compare only designs with integrated controllers:
  - `classical`
  - `pinnacle`
  - otherwise mark as external / not modeled here
- If scratchpad area or bandwidth is discussed, keep total `scratchpad + accumulator` memory capacity fixed and compare interface style rather than just module presence.
- Consider adding a small unifying instruction / controller shell for:
  - `onesa`
  - `systolicattention`
  - optionally `fusemax`
    so full-system comparison uses more consistent sequencing boundaries
- Consider adding one "comparison contract" markdown note for all baselines defining:
  - compare-at-PE scope
  - compare-at-array scope
  - compare-at-datapath-wrapper scope
  - compare-at-system scope
  - what is excluded at each scope

## Bottom line

Current best-practice comparison:

- Compare all five at compute-core scope.
- When comparing ONE-SA to other arrays, normalize for its SIMD MAC inside each PE.
- Compare all five at accumulator-compute-unit scope under fixed total `scratchpad + accumulator` memory capacity.
- Compare `classical` and `onesa` directly for minimal-accumulator datapath structure.
- Compare `classical` and `pinnacle` directly for full-system wrapper structure.
- Treat FuseMax as `2D SA + 1D accumulator`.
- Treat `systolicattention` as a specialized architecture whose structural wrapper is now closer to the others than before, but whose controller boundary and accumulator semantics still differ too much for naive one-to-one full-system comparison.
