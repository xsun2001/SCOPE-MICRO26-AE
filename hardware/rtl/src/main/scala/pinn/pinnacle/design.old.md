# PINNACLE Systolic Array Design

This note documents the current implementation in [`sa.scala`](${SOURCE_ROOT}/rtl/src/main/scala/pinn/pinnacle/sa.scala), following [`design.template.md`](${SOURCE_ROOT}/rtl/src/main/scala/pinn/design.template.md).

The file contains three layers:

- a modified `SystolicArray` that supports both classical GEMM and strip-mode PINN evaluation,
- a generic ping-pong `Accumulator`,
- a tile-scoped `PinnacleAttentionTileSystem` wrapper that maps one FlashAttention tile onto the modified array.

## What the difference vs classical SA

Compared with [`rtl/src/main/scala/pinn/classical/sa.scala`](${SOURCE_ROOT}/rtl/src/main/scala/pinn/classical/sa.scala), the PINNACLE design keeps the original weight-stationary GEMM path but adds a second datapath for evaluating

$$
f(x) = \sum_{i=0}^{H-1} \mathrm{ReLU}(w_i x + b_i)
$$

inside selected strip rows.

The concrete architectural differences are:

- `SystolicArrayOp` grows from `LOAD_WEIGHT, COMPUTE` to `LOAD_WEIGHT, COMPUTE, LOAD_PINN, PINN, PINN_DRAIN`.
- `SystolicArrayIO.op` is a per-row vector instead of one global opcode. This is required because strip rows and non-strip rows can execute different ops in the same cycle.
- `PEMesh` can reserve the first `stripHeight * stripCount` rows as strip rows. Those rows are built from `PELeft`/`PERight` pairs instead of plain MAC PEs.
- A strip row has a second north-input path, `in_n_strip`, used only at strip top rows. This lets the wrapper inject `x`, `w`, and `m - m_new` without disturbing the classical deskewed north stream.
- The array exposes two outputs:
  - `out_s`: the normal deskewed bottom-row GEMM result,
  - `out_s_raw`: raw taps from the bottom row of each strip region.
- During `PINN`, strip rows reuse the west stream to preload `V` into `reg_weight` while the strip engine evaluates `exp` approximations. The classical design has no equivalent overlap.
- The wrapper fuses `QK`, strip coefficient reload, `P = exp(S - m_new)`, `alpha = exp(m - m_new)`, and `V` preload into one controller state, `RunFused`.

Additional hardware vs the classical array:

- `PELeft` adds one extra resident register, `reg_pinn_weight`.
- Each strip pair adds two local side-channel links:
  - `PERight.out_left_b -> PELeft.in_right_b`,
  - `PELeft.out_right_p -> PERight.in_left_p`.
- The mesh adds strip-top input muxing and strip-boundary result taps.
- The wrapper adds strip-output deskewing and row assembly logic.

Supported operations beyond classical GEMM:

- `LOAD_PINN`: load strip coefficients from the north path.
- `PINN`: evaluate the strip approximator and simultaneously allow `V` preload through the west path.
- `PINN_DRAIN`: continue draining strip results while holding the reused GEMM weights stationary.

Area/power overhead:

- There is no synthesis report in this repo for an exact number.
- From the RTL structure, the overhead is localized to strip rows and wrapper control.
- Static area increases mainly from one extra register per `PELeft`, the pair-local routing, extra muxing, and the additional strip-output alignment logic.
- Dynamic power overhead is only exercised during PINN phases; pure GEMM mode still uses the original MAC path.

## Description of 2D PE

The 2D mesh contains two PE organizations:

- `PEStandard` for normal rows,
- `PELeft` + `PERight` pairs for strip rows.

### `PEStandard`

`PEStandard` is the classical weight-stationary MAC PE.

```text
                 reg_weight
                     ^
                     |
west in_w ----> [ reg_e ] ----------------------> east out_e
                  |
                  |                +---------------------------+
north in_n ------>|--------------->| reg_s := in_n + w * in_w |--> south out_s
                                   +---------------------------+
```

Operation behavior:

- `LOAD_WEIGHT`
  - `reg_e := in_w`
  - `reg_s := in_n`
  - `reg_weight := in_w`
- `COMPUTE`
  - `reg_e := in_w`
  - `reg_s := in_n.mac(reg_weight, in_w)`
- `LOAD_PINN`, `PINN`, `PINN_DRAIN`
  - no case is defined in `PEStandard`,
  - all registers hold their previous values.

That hold behavior is intentional. It lets non-strip rows stay quiescent while strip rows evaluate the approximator.

### Strip-row pair: `PELeft` + `PERight`

Each strip row is built from alternating left/right PEs. One pair implements one ReLU term and one running vertical accumulation lane.

```text
                 north even lane: x or w
                         |
                         v
                 +-------------------+     affine p = b + w * x     +-------------------+
west ----------->| PELeft            |------------------------------>| PERight           |----------> east
V preload / pass | reg_pinn_weight=w |                               | reg_e holds b     | V preload / pass
                 | reg_e holds x     |<------------------------------| out_left_b = b    |
                 | reg_s holds p     |            bias b             | reg_s holds sum    |
                 +-------------------+                               +-------------------+
                         |                                                     |
                         | x pipeline                                          | accumulated sum
                         v                                                     v
                    south even lane                                      south odd lane / strip tap
```

The pair-local links are:

- `PERight.out_left_b -> PELeft.in_right_b`: provides the resident bias to the affine stage.
- `PELeft.out_right_p -> PERight.in_left_p`: provides the affine result to the ReLU-and-accumulate stage.

The vertical meaning depends on mode:

- in classical mode, both PEs behave like normal systolic MAC cells,
- in strip mode, the left PE forwards `x` downward and the right PE forwards the running strip sum downward.

Operation behavior:

#### `LOAD_WEIGHT`

Both `PELeft` and `PERight` act like classical MAC PEs:

- forward `in_w` to `reg_e`,
- forward `in_n` to `reg_s`,
- capture `reg_weight := in_w`.

This keeps strip rows compatible with normal GEMM usage.

#### `COMPUTE`

Both `PELeft` and `PERight` perform classical MAC:

- `PELeft.reg_s := io.in_n.mac(reg_weight, io.in_w)`
- `PERight.reg_s := io.in_n.mac(reg_weight, io.in_w)`

So strip rows can also participate in `QK` and `PV`.

#### `LOAD_PINN`

This loads strip coefficients from the north side:

- `PELeft.reg_pinn_weight := io.in_n.withWidthOf(dataType)` stores `w_i`.
- `PERight.reg_e := io.in_n.withWidthOf(dataType)` stores `b_i`.

The wrapper supplies coefficient rows as alternating `[w, b, w, b, ...]`, so even columns feed left PEs and odd columns feed right PEs.

#### `PINN`

`PINN` uses the strip datapath:

- `PELeft` computes the affine stage

$$
p_i = b_i + w_i x
$$

  using `reg_pinn_weight` and the paired bias from `PERight.out_left_b`.

- `PELeft` also forwards the previous `x` sample vertically:

$$
x_{\text{south}}(t+1) = x_{\text{north}}(t)
$$

- `PERight` computes the vertical accumulation stage

$$
\text{sum}_{\text{south}} = \text{sum}_{\text{north}} + \mathrm{ReLU}(p_i)
$$

- both PEs simultaneously latch `reg_weight := io.in_w`, so the west stream can preload `V` while the strip engine is running.

#### `PINN_DRAIN`

`PINN_DRAIN` uses the same strip arithmetic as `PINN`, but does not overwrite `reg_weight`.

This has two purposes:

- continue draining the strip pipeline,
- preserve the `V` weights already loaded during `PINN`.

### Strip top-row injection and strip output taps

Only the top row of each strip region overrides its normal north input:

- on `LOAD_PINN`, the strip top row sees `in_n_strip` delayed by the row index,
- on `PINN` or `PINN_DRAIN`, left PEs receive packed `x` values and right PEs receive zero on their north input.

At the bottom row of each strip, the right PE output is tapped into `out_s_raw`. The wrapper later deskews these taps into a logical output row.

## Description of 1D PE/Accumulator

`Accumulator` is a 1D ping-pong vector accumulator backed by a two-bank SRAM.

```text
                 +----------------------------------------------+
SA in ---------->| write pipeline                              |
                 |                                              |
                 |  rd bank A ---> old psum ----+               |
                 |                              |               |
                 |      WRITE: use input -------+--> wr bank A  |
                 |      ACCUM: psum + input ----+               |
                 |                                              |
consumer out <---| rd bank B                                   |
                 +----------------------------------------------+
                           ^
                           |
                        flip bank
```

Interface behavior:

- `WRITE`
  - store the incoming line at the current `in_idx`,
  - ignore the old psum.
- `ACCUM`
  - read the old psum from the active bank,
  - write back `psum + in`.
- `flip`
  - swap the ping and pong bank roles.
- `in_en`
  - capture a pending write and increment `in_idx`.
- `out_en`
  - increment `out_idx`.

Important status of the current wrapper:

- `Accumulator` is implemented in this file,
- `PinnacleAttentionTileSystem` does not currently instantiate it,
- `l_new` and `O_new` are updated directly in wrapper registers instead.

So the accumulator is part of the design space, but not yet on the active FlashAttention tile datapath in this file.

## Summary of register usage

All PEs are fully registered. A value written in cycle `t` is observed on the registered output in cycle `t + 1`.

### `PEStandard`

| Register | Type | Purpose | `LOAD_WEIGHT` | `COMPUTE` | `LOAD_PINN` / `PINN` / `PINN_DRAIN` |
| --- | --- | --- | --- | --- | --- |
| `reg_e` | `dataType` | East-going activation pipeline register | `in_w` | `in_w` | hold |
| `reg_s` | `accType` | South-going psum register | `in_n` | `in_n + reg_weight * in_w` | hold |
| `reg_weight` | `dataType` | Resident GEMM weight | `in_w` | hold | hold |

### `PELeft`

| Register | Type | Purpose | `LOAD_WEIGHT` | `COMPUTE` | `LOAD_PINN` | `PINN` | `PINN_DRAIN` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reg_e` | `dataType` | Classical east register or strip `x` pipeline latch | `in_w` | `in_w` | hold | `in_n.withWidthOf(dataType)` | `in_n.withWidthOf(dataType)` |
| `reg_s` | `accType` | Classical psum or strip affine result | `in_n` | `in_n + reg_weight * in_w` | `in_n` | `in_right_b + reg_pinn_weight * in_n` | `in_right_b + reg_pinn_weight * in_n` |
| `reg_weight` | `dataType` | Resident GEMM weight or overlapped `V` preload storage | `in_w` | hold | hold | `in_w` | hold |
| `reg_pinn_weight` | `dataType` | Resident strip coefficient `w_i` | hold | hold | `in_n.withWidthOf(dataType)` | hold | hold |

### `PERight`

| Register | Type | Purpose | `LOAD_WEIGHT` | `COMPUTE` | `LOAD_PINN` | `PINN` | `PINN_DRAIN` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reg_e` | `dataType` | Classical east register or resident strip bias `b_i` | `in_w` | `in_w` | `in_n.withWidthOf(dataType)` | hold | hold |
| `reg_s` | `accType` | Classical psum or strip running sum | `in_n` | `in_n + reg_weight * in_w` | `in_n` | `in_n + relu(in_left_p)` | `in_n + relu(in_left_p)` |
| `reg_weight` | `dataType` | Resident GEMM weight or overlapped `V` preload storage | `in_w` | hold | hold | `in_w` | hold |

### `Accumulator`

| Register | Type | Purpose | Update |
| --- | --- | --- | --- |
| `ping_pong` | `Bool` | Select which half of SRAM is write-active | Toggles on `flip` |
| `in_idx` | `UInt` | Input row pointer | Incremented on `in_en` with wraparound |
| `out_idx` | `UInt` | Output row pointer | Incremented on `out_en` with wraparound |
| `writePending` | `Bool` | One-cycle delayed SRAM write enable | Set from `io.in_en` |
| `writeAddr` | `UInt` | Latched write address | Captured when `in_en` is high |
| `writeData` | `Vec(N, accType)` | Latched write payload | Captured when `in_en` is high |
| `writeOp` | `AccumulatorOp` | Distinguish `WRITE` from `ACCUM` | Captured when `in_en` is high |

## Dataflow of a full flashattention tile

The wrapper targets the standard online-softmax recurrence:

$$
S = QK^T \\
m_{\text{new}} = \max(m, \mathrm{rowmax}(S)) \\
P = \exp(S - m_{\text{new}}) \\
\alpha = \exp(m - m_{\text{new}}) \\
l_{\text{new}} = \alpha \cdot l + \mathrm{rowsum}(P) \\
O_{\text{new}} = \alpha \cdot O + PV \\
m = m_{\text{new}}
$$

In the current implementation, `exp` is approximated by the strip engine:

$$
\exp(x) \approx \sum_{i=0}^{H-1} \mathrm{ReLU}(w_i x + b_i)
$$

with:

- `H = stripHeight`,
- `stripCount in {1, 2}`,
- `segmentsPerRow = 2 / stripCount`,
- `saLatency = 2N - 1`,
- aligned strip retire latency `N + 1`,
- raw strip-piece latency for strip `s`: `(s + 1)H + 1`.

### Step-by-step schedule

#### 1. Scratchpad initialization

State: `InitIssue`, `InitWait`

The wrapper reads one tile of:

- coefficient rows: `stripHeight`,
- `Q`: `N` rows,
- `K^T`: `N` rows,
- `V`: `N` rows,
- `O`: `N` rows,
- `m`: `1` row,
- `l`: `1` row.

Each read takes two controller cycles, so initialization costs:

$$
T_{\text{init}} = 2(\text{stripHeight} + 4N + 2)
$$

#### 2. Preload `K^T`

State: `PreloadKt`

For `N` cycles, the wrapper drives:

- `saOp = LOAD_WEIGHT` on all rows,
- `saWest(row) = ktRows(row)(N - 1 - preloadCol)`.

After this phase, all PEs hold the `K^T` tile in `reg_weight`.

Cost:

$$
T_{K^T\text{ preload}} = N
$$

#### 3. Compute `S = QK^T` and update `m_new`

State: `RunFused`, early part

For cycles `0 .. N - 1` of `RunFused`, the wrapper injects `Q` rows with `COMPUTE`.

- Row `r` of `Q` is launched at fused-cycle `r`.
- The aligned GEMM result row appears at `out_s` after `saLatency = 2N - 1` cycles.
- When row `r` retires, the wrapper stores:
  - `sRows(r) := S(r, :)`
  - `mNewRows(r) := max(mRows(r), rowmax(S(r, :)))`

The array is still a standard weight-stationary GEMM engine during this phase.

#### 4. Reload strip coefficients

State: still `RunFused`

At fused-cycles `2N .. 2N + stripHeight - 1`, the wrapper injects coefficient rows with `LOAD_PINN`.

This phase is diagonalized per strip row:

- the top row of a strip loads all `stripHeight` coefficient rows,
- deeper rows load fewer cycles,
- after a row has loaded its intended coefficient, it switches to `PINN_DRAIN` so it is not overwritten by later coefficient rows.

This coefficient reload overlaps with the retiring tail of `QK`, because the last `S` rows are still draining from the classical array while the strip rows are being configured.

#### 5. Compute `P = exp(S - m_new)` while preloading `V`

State: still `RunFused`

As soon as both conditions hold:

- strip coefficients are loaded,
- `sRows(row)` is available,

the wrapper starts issuing strip launches.

For each logical row:

- north input gets `S(row, :) - mNewRows(row)`,
- strip rows run `PINN` or `PINN_DRAIN`,
- non-strip rows run `LOAD_WEIGHT` only when a `V` preload should occur.

`stripCount` determines how many launches are needed per logical row:

- `stripCount = 2`
  - one strip launch produces one full `N`-lane `P` row.
- `stripCount = 1`
  - two strip launches are needed,
  - each launch produces `N / 2` outputs,
  - the wrapper packs them into even lanes on input and unpacks them back into one logical row on output.

Important overlap:

- when the last segment of a logical `P` row is issued, the west path simultaneously preloads one `V` column into `reg_weight`,
- strip rows use `PINN` for this overlap,
- non-strip rows use `LOAD_WEIGHT`.

So there is no dedicated `V` preload state after the strip phase.

#### 6. Compute `alpha = exp(m - m_new)`

State: still `RunFused`

After all `P` launches have been issued, the wrapper reuses the same strip engine for:

$$
\alpha = \exp(m - m_{\text{new}})
$$

The input is:

$$
mRows - mNewRows
$$

and the strip rows run `PINN_DRAIN`.

`PINN_DRAIN` is used instead of `PINN` so the already-loaded `V` weights are preserved for the next GEMM phase.

The wrapper uses the dedicated `in_n_strip` stream for strip-only work, so `alpha` launch no longer pollutes the
classical top-row psum input. After the final `alpha` segment is injected, the external controller can return to
classical `COMPUTE` on the very next cycle.

Because both `saOp` and the west input are already skewed by physical row index, row `r` sees:

- `PINN_DRAIN` for `alpha` at its strip-use cycle,
- then `COMPUTE` one cycle later when the `PV` wavefront reaches that same row.

So `PV` launch overlaps the strip-side `alpha` drain itself. The wrapper only needs `alphaRows` to be ready by the time
`pvOutValid` retires a row, not by the time that row is launched into `PV`.

#### 7. Form `l_new`

Whenever a strip output retires:

- if it is a `P` row, the wrapper stores the row in `pRows` and accumulates `rowsum(P)`,
- if it is an `alpha` row, the wrapper stores it in `alphaRows`.

Once both are available for a logical row, the wrapper computes:

$$
l_{\text{new}} = \alpha \cdot l + \mathrm{rowsum}(P)
$$

This happens in wrapper registers, not in `Accumulator`.

#### 8. Compute `PV` and form `O_new`

State: `RunPV`

The wrapper now returns to classical GEMM mode:

- `saOp = COMPUTE`,
- west input streams ready `P` rows as soon as `pRows(pvLaunchIdx)` has been assembled,
- resident `V` weights already stored in `reg_weight` produce `PV`.

`RunPV` starts as soon as:

- all strip launches have been issued,
- row `0` of `P` is available.

It does **not** wait for `alpha` to be fully assembled or for the strip mesh to finish draining. Those events continue in
parallel with the `PV` wavefront.

Row `r` of `P` is launched once, and after `saLatency = 2N - 1` cycles its `PV` row retires. The wrapper then forms:

$$
O_{\text{new}}(r, :) = \alpha(r) \cdot O(r, :) + PV(r, :)
$$

again in wrapper registers.

#### 9. Write back results

State: `Writeback`, then `Done`

The wrapper writes back:

- `O_new`: `N` rows,
- `m_new`: `1` row,
- `l_new`: `1` row,
- then pulses `done`.

### Overlap summary

The main pipeline overlaps are:

- `QK` tail overlaps strip coefficient reload.
- `P` issue overlaps `V` preload.
- `alpha` issue overlaps the drain of late strip results.
- `PV` launch overlaps both the old wrapper-side strip deskew tail and the in-array `alpha` drain; after the last `alpha` segment is injected, the controller immediately returns to classical `COMPUTE`.
- `PV` does not require an extra `V`-load phase because `V` was already captured during `P` issue.

### Tile latency

The directed testbench [`PinnacleAttentionTileTest.scala`](${SOURCE_ROOT}/rtl/src/test/scala/pinn/PinnacleAttentionTileTest.scala) checks the exact `done` latency:

$$
T_{\text{tile}} = 16N + 3 \cdot \text{stripHeight} + 8 + \delta_{\text{full}} + (2 - \text{stripCount})(N + 1)
$$

where

$$
\delta_{\text{full}} =
\begin{cases}
1 & \text{if } \text{stripCount} = 2 \text{ and } \text{stripHeight} \cdot \text{stripCount} = N \\
0 & \text{otherwise}
\end{cases}
$$

Because `stripCount in \{1, 2\}`, this simplifies to:

- `stripCount = 2` and `2 \cdot stripHeight < N`

$$
T_{\text{tile}} = 16N + 3 \cdot \text{stripHeight} + 8
$$

- `stripCount = 2` and `2 \cdot stripHeight = N`

$$
T_{\text{tile}} = 16N + 3 \cdot \text{stripHeight} + 9
$$

- `stripCount = 1`

$$
T_{\text{tile}} = 17N + 3 \cdot \text{stripHeight} + 9
$$

Removing wrapper-side scratchpad init `2(H + 4N + 2)` and the writeback/done tail `N + 3` gives the compute-only
preload-to-`PV` cost:

$$
T_{\text{compute}} = 7N + \text{stripHeight} + 1 + \delta_{\text{full}} + (2 - \text{stripCount})(N + 1)
$$

So for the common `stripCount = 2` configuration, the tile core is `7N + H + 1` for partial-strip layouts and
`7N + H + 2` for the full-strip layout `2H = N`. Both are in the `7N + O(1)` regime when `H` is treated as a fixed
approximator depth.

Compared with the previous barrierized controller, the streamed `alpha -> PV` handoff removes the dedicated alpha-drain
wait from the critical path. The exact single-tile savings are:

- `stripCount = 2`: `2H - 1`
- `stripCount = 1`: `H`

### Full-size FlashAttention kernel

`PinnacleAttentionTileSystem` is only a single-tile engine. It does not implement the outer loops over all query blocks and key/value blocks of a full attention kernel.

So the only exact cycle number defined by this file is `T_tile` above.

If a higher-level controller tiled a larger problem into `T_q` query tiles and `T_k` key/value tiles with no inter-tile overlap, the compute-side lower bound would be:

$$
T_{\text{full}} \approx T_q \cdot T_k \cdot T_{\text{tile}}
$$

For a square sequence length `L` and tile size `N`, that becomes:

$$
T_{\text{full}} \approx \left\lceil \frac{L}{N} \right\rceil^2 T_{\text{tile}}
$$

This is only an extrapolation. Exact full-kernel latency would also depend on outer-loop control, scratchpad refill policy, and any overlap between tiles, none of which are implemented in this file.
