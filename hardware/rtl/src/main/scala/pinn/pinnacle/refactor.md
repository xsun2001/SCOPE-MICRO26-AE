# Task

Strictly follow this task list:

1. Enter rtl/src/main/scala/pinn/pinnacle/ directory.
2. Read design.md for full design document.
3. Read refactor.md for refactor plan
4. Read current source files and draft todo list. Ignore any .old files.
5. Implement the plan step by step.

# Refactor Plan

1. Define the architectural contract in sa.scala.
   - Keep one source of truth for legal config: N even, S > 0, and 2S <= N. The active RTL is two-strip only.
   - Separate steady-state throughput timing from full-retirement timing in constants and comments so the controller never conflates them.
   - Define instruction and micro-op enums for:
     - SA west path: idle, gemm preload, gemm compute, pinn coeff, optional preload during pinn compute.
     - SA north strip path: idle, pinn data.
     - ACC ops: tile store, tile load-affine, tile load-restream, scaled tile store, max reduction, sum reduction, vector reset, vector commit.
2. Rebuild PE behavior to match the design document exactly.
   - PEStandard: unchanged GEMM datapath with reg_e, reg_s, reg_w.
   - PELeft: add explicit reg_x and clean separation between GEMM state and Pinn state.
     - pinn.setup loads w_i through the west path.
     - pinn.compute uses the MAC as b + w\*x, forwards x downward via reg_x, and emits p to the paired right PE.
   - PERight: keep one MAC and ReLU datapath.
     - pinn.setup loads b_i through the west path.
     - pinn.compute performs y + ReLU(p) using the MAC/add reuse path.
   - Preserve the design’s reinterpret-cast rule when x travels in the accType north/south path.
3. Rebuild the strip and mesh around explicit pair/lane structure.
   - Model neighboring columns (2j, 2j+1) as one fixed Pinn lane.
   - Reserve top S rows for strip 0 and next S rows for strip 1.
   - Implement the two-strip logical-lane routing: logical element i goes to strip i % 2.
   - Keep standard systolic skew buffers for GEMM west/north traffic.
   - Expose two south outputs:
     - Bottom GEMM drain for normal SA output.
     - Strip drain for Pinn results with enough metadata to associate row/piece timing in the controller.
4. Redesign the accumulator to match the spec instead of extending the current ping-pong buffer.
   - Use N banks of SyncReadMem, element-granular storage, one element per bank per cycle.
   - Store O and T tiles in SRAM.
   - Hold M0, M1, alpha, L, rowsum as five dedicated vector registers, not SRAM-backed tiles.
   - Support these primitives:
     - Store tile row with max reduction into M1.
     - Load tile row with vector subtraction for S - M1.
     - Store Pinn output row into T.
     - Load T row both to west restream and to rowsum reduction.
     - Store PV row into O with row-wise scale by alpha.
     - Commit M1 -> M0.
     - Reset vector state on attn.setup.
   - Ignore internal hazards, as allowed by the spec, but make controller scheduling explicit enough that illegal conflicts are structurally avoided.
5. Add a new top-level Pinnacle tile system in sa.scala.
   - Include:
     - instruction port
     - scratchpad host port
     - optional ACC host/debug port if still useful for tests
     - SA
     - ACC
     - scratchpad datapath muxing
     - fixed-latency controller
   - Supported milestone instructions:
     - gemm.preload
     - gemm.compute
     - pinn.setup
     - pinn.compute
     - attn.setup
     - attn.compute
   - Exclude attn.rescale completely for now.
6. Implement attn.setup as a pure state/setup instruction.
   - Latch accO, accT, spCoeff.
   - Reset controller sequencing state.
   - Reset vector registers to:
     - M0 = -inf
     - M1 = -inf
     - alpha = 0
     - L = 0
     - rowsum = 0
   - Do not add extra tile-sized state beyond O/T SRAM and the five required vectors.
7. Implement attn.compute as a fixed microsequence matching the design schedule.
   - Stage A, 0..N: preload K.
   - Stage B, N..2N: GEMM QK.
   - Stage C, 2N..3N: pinn.setup coeff load while ACC stores S and reduces rowmax into M1.
   - Stage D, 3N..4N: pinn.compute for P = exp(S - M1) while preloading V.
   - Stage E, delayed by S+1: ACC stores P into T.
   - Stage F, 4N..5N: ACC reloads P to SA west for PV, and concurrently reduces rowsum(P).
   - Stage G, 5N..7N: ACC stores O = alpha \* O + PV.
   - Stage H, retirement tail: commit M1 -> M0, finalize any outstanding L updates, and gate instruction completion.
8. Encode the completion policy directly in the controller.
   - If the next queued instruction is attn.compute, allow reissue at:
     - 5N+2 in two-strip mode
   - If the next instruction is not attn.compute, hold busy until full tile retirement at:
     - 7N+2 in two-strip mode
   - This means the controller needs:
     - one notion of pipeline-front availability
     - one notion of architectural completion
   - Tests should assert both behaviors separately.
9. Keep the controller simple by making the schedule compile-time fixed.
   - No dynamic stalls once issued.
   - No scoreboard beyond minimal “next instruction is also attn.compute” gating.
   - No extra vector/tile shadow registers.
   - All row indices, piece ids, and delayed valid bits should be generated from counters and fixed delay lines.
10. Restore the testbench in increasing-risk order.

- GEMM test:
  - preload + compute correctness
  - exact N preload + N inject + N drain timing
- Pinn strip unit/system test:
  - coeff load correctness
  - S+1 first-result latency
  - two-strip full-throughput behavior
- ACC test:
  - rowmax reduction into M1
  - affine reload S - M1
  - rowsum reduction from P
  - scaled O update with alpha
  - M1 -> M0 commit
- Single-tile attention test:
  - result correctness
  - 7N+2 or 8N+2 completion when followed by non-attention work
- Back-to-back attn.compute test:
  - second issue accepted at 5N+2 or 6N+2
  - final architectural results still match fully retired behavior
- Mixed-instruction barrier test:
  - attn.compute followed by gemm.\* must not start early
- Matrix of default regressions:
  - S = 2, 4, 8
  - N = 4, 8
  - skip invalid configs
  - dataType = sint16, accType = sint32

# Recommended Execution Order

1. PE and mesh rewrite.
2. ACC rewrite.
3. Top-level controller and instruction shell.
4. GEMM tests.
5. Pinn tests.
6. Single-tile attention test.
7. Back-to-back attention overlap test.
8. Full regression across valid (N, S) pairs.

In rtl/src/main/scala/pinn/pinnacle
Strictly follow the design.md documented register usage and datapath.
1. PELeft should have 4 reg, and PERight should have 3 reg. Please enforce the register reuse.
2. Do not introduce dedicated `StripDataIn/Out`. You should reuse those IO port/data path. There should only have north south west easy direction, without `data` or `strip` difference.
3. The current version are adding too much unexpected hardware overhead and severely shift from our motivation. FIX THEM AS FAST AS POSSIBLE.
4. You only need to study design.md and sa.scala. DO NOT REFER TO .old FILES.
