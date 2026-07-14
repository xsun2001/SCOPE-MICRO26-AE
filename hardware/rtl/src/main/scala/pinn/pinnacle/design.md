# PINNacle Systolic Array design

## Key principle

- We want to fuse nonlinear function approximator into the systolic array with minimal hardware overhead.
  - Neuron approximator: $y(x) \approx \sum_i^S \mathrm{ReLU}(w_i \cdot x + b_i)$. We call it `Pinn` for the following document.
  - The apprximator is handled by $N x S$ (width x height) PE strip, splitted to $N/2$ vertical lanes. Each lane contains $S$ PE pair. The left PE handles $p = w_i \cdot x + b_i$, and the right PE handles $y' = y + \mathrm{ReLU}(p)$ in the next cycle. MAC units can be reused. $S+1$ latency
  - Each strip has $N/2$ vertical lanes, providing $N/2$ elements per cycle throughput. The active RTL reserves two strips stacked vertically, so the legal geometry rule is $2S \leq N$ and the nonlinear path keeps full $N$ elements per cycle throughput with the same $S+1$ latency.
  - Two strips provide the intended fully fused flashattention throughput with $5N+2$ cycles per tile (steady pipeline).
  - Strip height $S$ is typically 4, 8, 16. $N$ is from 8, 16, 32, 64, 128.
  - Default legal config rule: `N` must be even, `S > 0`, and `2S <= N`.
  - Latency note: the quoted `S+1` for Pinn means first-input to first-output pipeline latency of one strip. Steady-state throughput is `N/2` elements per cycle per strip, and the active two-strip RTL combines both strips for full-width issue.
  - Data format: support commonly adopted types, including `sint8/16/32` and `fp8/16/32`. This is a generic modification to systolic attention. `dataType` is the horizontal storage/stream type, and `accType` is the vertical accumulation/storage type. Typically `accType` is wider than `dataType`.
- Design requirements:
  - Minimal hardware overhead. Unless we **explicitly** claim in this design document, DO NOT introduce any additional vector- or tile-size register; DO NOT introduce any additional heavy compute hardware (e.g. ALU with large bitwidth).
  - Simple controller. Additional to normal systolic array, controller should be a simple micro-op sequencer with fixed latency and order. No more
  - System module. One large standard scratchpad (SP). One accumulator (ACC) with vector ALU to support reduction, scale, pre- and post-process. Datapath: SP/ACC -> SA -> ACC -> SP.
- Supported Instructions
  - `gemm.preload` spW
  - `gemm.compute` spA (or accA)
  - `pinn.setup` spCoeff
  - `pinn.compute` spX (or accX)
  - `attn.setup`: accO, accT, spCoeff
  - `attn.compute`: spQ, spK, spV
  - `attn.rescale`: spO (rescale and load back to scratchpad)
  - Note:
    - Max arg count is three. Reuse the arg slots. sp is for scratchpad address. acc is for accumulator address.
    - Default instruction bundle is `opcode + 4 address args`.
    - ACC address: Tile O, T (for S, P). Tile size $N\times N$. Tiles are row-major in memory.
    - `M0`, `M1`, `alpha`, `L`, `rowsum` are five dedicated vector registers of length $N$, not part of ACC SRAM.
    - `attn.setup` only latches `accO`, `accT`, `spCoeff` base addresses and resets controller state. It also resets the five vector registers.
    - Default init values for the first attention tile: `O=0`, `T` don't care, `M0=-inf`, `M1=-inf`, `alpha=0`, `L=0`, `rowsum=0`.
    - `attn.compute` is blocking, fixed-latency, and does not stall internally once issued. It executes the steady pipeline in `5N+2` cycles in the active two-strip RTL.
    - `attn.rescale` computes the final normalization `O / L` and writes it back to scratchpad. It reuses the reciprocal coefficients from `spCoeff` set by `attn.setup`.

## `sa.scala` Core systolic array design

### Process Element

- Naming: mostly directional. west, east, north, south => \_w,\_e, \_n,\_s. \_w somethings is for weight.
- PEStandard:
  - reg_e, reg_s, reg_w
  - Preload: in_w -> reg_w -> out_e
  - Compute: in_w -> reg_e -> out_e. in_n.mac(reg_w, in_w) -> reg_s -> out_s
- PELeft:
  - reg_e, reg_s, reg_w, reg_x
  - Preload: in_w -> reg_w -> out_e
  - Compute: in_w -> reg_e -> out_e. in_n.mac(reg_w, in_w) -> reg_s -> out_s
  - PinnCoeff: in_w -> reg_e -> out_e
  - PinnCompute: in_right_b.mac(in_n, reg_e) -> reg_s -> out_right_p, in_n -> reg_x -> out_s
- PERight:
  - reg_e, reg_s, reg_w
  - Preload: in_w -> reg_w -> out_e
  - Compute: in_w -> reg_e -> out_e. in_n.mac(reg_w, in_w) -> reg_s -> out_s
  - PinnCoeff: in_w -> reg_e -> out_e
  - PinnCompute: reg_e -> out_left_b, in_left_p -> ReLU -> + in_n -> reg_s -> out_s (+ reuse mac unit)
- Notes:
  - Reuse most of the registers. Only one MAC is needed for each PE. The hardware overhead, excepting datapath and muxes, only includes one `reg_x` in PELeft and one `ReLU` in PERight.
  - PinnCompute can overlap with preload. Their datapath and register usage are not interfere. Preload use horizontal dataflow and reg_w, while PinnCompute uses vertical dataflow and other registers.
  - In one lane, PELeft and PERight form a fixed local pair. `out_left_b`/`in_right_b` and `out_right_p`/`in_left_p` are intra-lane connections only and do not cross lanes.
  - PinnCoeff: `spCoeff` contains $S$ pairs of $w_i,b_i$. The $i$-th strip row is pushed by a scratchpad row that repeats the pair, e.g. for `N=4`, row 0 is `w_1, b_1, w_1, b_1`. To simplify this repeated load, each pinn strip row may have two extra small staging registers for `w_i` and `b_i`.
  - DataType: `reg_e`, `reg_w`, `reg_x` are `dataType`. `reg_s` is `accType`. MAC unit is `dataType x dataType + accType -> accType`. When PinnCompute, input `x` is `dataType` but streams through the `accType` datapath. Use the low `dataType`-width bits as a reinterpret cast instead of numeric conversion. For FP, treat those low bits as the original low-width FP payload during PinnCompute only.

### Systolic Array

- $N x N$ systolic array. Can handle normal preload and compute (GEMM) operations with exact timing.
- Datapath:
  - Normally, all data and op are skewed by input buffer for systolic dataflow. The outputs of normal SA and strip are short-circuited to ACC. As a result, SP has row-level access but ACC may need element-level fine-grained access.
  - In fused flashattention, the results of P=exp(S - rowmax(S)) should be restreamed into the west input of SA to perform GEMM compute O=PV. So a dedicated, short-circuited path of ACC should be maintained. Use two pointers for it, instead of using another vector register.
- Default physical mapping: lane `j` uses neighboring columns `(2j, 2j+1)` as one left/right PE pair. Strip rows occupy the top `S` rows of each strip.
- The top $N x 2S$ region contains two strip (if has enough space). The input elements are splitted by odd-even. The `i`-th element is routed to `i%2` strip for process. If only one strip exists, each logical row is processed in two cycles: even positions first, then odd positions in the next cycle.
- Systolic array ops
  - `gemm.preload`: West weight
  - `gemm.compute`: West activation
  - `pinn.setup`: West coeff
  - `pinn.compute` (it also includes `gemm.preload` function optionally): West preload weight, North pinn input x
  - Default one-tile GEMM timing: `N` cycles preload, `N` cycles compute injection, and `N` cycles drain/store to ACC.

### Accumulator

- ACC needs element-level access to support fast restreaming. Use it SRAM as output buffer to free register usage.
- ACC has `N` banks of chisel `SyncReadMem`, one element per bank per cycle, with one-cycle read latency. All ACC SRAM banks are `1R1W`.
- Attention state uses five dedicated vector registers: `M0`, `M1`, `alpha`, `L`, `rowsum`.
- Operations:
  - Store (Data -> ACC)
    - With reduction: max, sum (for rowmax, rowsum)
    - With scale: AccTile = ScaleVector \* AccTile + InputTile (for $O_{\text{new}} = \alpha \cdot O + PV$)
  - Load (Acc -> Data)
    - With affine: subtract one vector from output data. Used for $S - m_{\text{new}}$ and for the pre-exp value $m - m_{\text{new}}$.
    - With scale: OutputTile = ScaleVector \* AccTile (for final `O / L` writeback to scratchpad)
  - Special
    - Swap M0, M1 address (for $m = m_{\text{new}}$)
    - Update (for $l_{\text{new}} = \alpha \cdot l + \mathrm{rowsum}(P)$)
- Reuse mac unit as much as possible.
- ACC arithmetic is single-cycle once operands are present. Reduction, affine, scale and update add no extra streaming latency beyond the tile/vector movement.
- Default movement cost: one vector element per bank per cycle. Therefore one vector load/store takes `N` cycles, and one tile stream also takes `N` cycles.
- ACC hazards are ignored in this design doc. Assume the issued schedule never triggers illegal same-resource conflicts.

## `system.scala` System

### Components

- Systolic Array
- Accumulator
- 5 vector registers: `M0`, `M1`, `alpha`, `L`, `rowsum`
- Scratchpad (check rtl/src/main/scala/pinn/common/scratchpad.scala). All SP SRAM banks are `1R1W`.
- Controller
- Data path and mux

### IO

- One instruction port (`opcode + 4 address args`, one in-flight instruction at a time)
- One read/write port for access scratchpad data

### Fused FlashAttention

$$
S = QK^T \\
m_{\text{new}} = \max(m, \mathrm{rowmax}(S)) \\
P = \exp(S - m_{\text{new}}) \\
\alpha = \exp(m - m_{\text{new}}) \\
l_{\text{new}} = \alpha \cdot l + \mathrm{rowsum}(P) \\
O_{\text{new}} = \alpha \cdot O + PV \\
m = m_{\text{new}}
$$

- The cycle-exact operations and dataflows for the top-left pe, or pe(0, 0). The other PEs can be inferred and execute same operation with delays.
  - Cycle 0: West: Preload K starts
  - Cycle N: West: GEMM S=QK starts
  - Cycle 2N: West: PinnCoeff starts; South: Results S write to T (tmp) tile in ACC (accumulator)
  - Cycle 3N: West: PinnCoeff ends; South: The first row of S and the first element of rowmax(S) are computed. `m - m_new` is also computed and saved in vector register `alpha`.
  - Cycle 3N+1: West: Preload V starts; North: ACC restreams S with minus rowmax(S) into north to compute Pinn exp.
  - Cycle 3N+1+S: South: Results P=exp(S-rowmax) start writing to T tile.
  - Cycle 4N: ACC starts reducing `rowsum(P)` while loading `P` back to SA for `PV` compute.
  - Cycle 4N+2: North: alpha = exp(m - m_new)
  - Cycle 4N+2: West: The first row of preload V ends. P=exp(S-rowmax(S)) can stream into west to compute O=PV.
  - Cycle 7N+2: Finish O=PV. Results are fully written to O tile.
  - Cycle 5N+2: Can pipeline next tile preload K. Pipeline steady latency 5N+2, finalize latency 2N.

| Cycle range     | Systolic array op                                             | Accumulator op                                                                 | Notes                                        |
| --------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| `0–N`           | `gemm.preload`: preload `K`                                   | `idle`                                                                         |                                              |
| `N–2N`          | `gemm.compute`: compute `QK`                                  | `idle`                                                                         |                                              |
| `2N–3N`         | `pinn.setup`: load coeff for `exp`                            | `store` with max reduction: store `S = QK` and compute `rowmax(S)`             |                                              |
| `3N–4N`         | `pinn.compute`: compute `P = exp(S - rowmax(S))`; preload `V` | `load` with affine: load `S - rowmax(S)` to SA                                 |                                              |
| `3N+S+1–4N+S+1` | —                                                             | `store`: store `P = exp(S - rowmax(S))` to `T`                                 | Starts after previous `load` by `S+1` cycles |
| `4N–5N`         | `gemm.compute`: compute `PV`                                  | `load` with sum reduction: load `P` to SA and compute `rowsum(P)` concurrently |                                              |
| `5N–7N`         | —                                                             | `store` with scale: store `O = \alpha \cdot O + PV`                            |                                              |

Note: The O(1) extra latency for \alpha, l is ignored for clarity in this table.

- This detailed table matches the active two-strip RTL.

### Final Rescale

- `attn.rescale` computes the final normalized output `O / L` and writes it back to `spO`.
- `1/x` is treated as another nonlinear function approximated by pinn mode, using the reciprocal coeff already pointed by `spCoeff`.
- This mode is specified for completeness only. The current RTL milestone will not implement `attn.rescale`, and the default testbench will not test it.
- Timeline:
  - `0-N`: `pinn.setup` for reciprocal `1/x`
  - `N-(N+S+1)`: `pinn.compute` on vector register `L`, producing vector `1/L`
  - `(N+S+1)-(2N+S+1)`: ACC loads `O`, applies `1/L` as row scale, and writes the normalized result to `spO`
- Total latency is `2N+S+1`.
- This mode is vector-only. Input and output do not need systolic skew buffers.

## Test -> `pinnacle.scala`

- $S=2,4,8$ $N=4,8$ (ignore invalid config)
- data type sint16, acc type sint32
- Other common types (`sint8/32`, `fp8/16/32`) are intended to be supported by the design, but are not covered by the default regression tests.
- Test cases
  - Normal GEMM (preload + compute)
  - One tile Fused FlashAttention (check result value and expected latency)
  - Two tiles Fused FlashAttention (check inter-tile overlapping)
  - `attn.rescale` is not implemented in the current RTL milestone and is not included in the default testbench.
