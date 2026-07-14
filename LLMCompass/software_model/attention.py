from hardware_model.device import Device
from software_model.matmul import Matmul, BatchedMatmul
from software_model.operators import Operator, Reshape, Concat, Transpose
from software_model.softmax import Softmax
from software_model.utils import Tensor, DataType, data_type_dict
from copy import deepcopy
from math import ceil, log2
import os


INT_SOFTMAX_MICROBENCHMARK = {
    # H100 Triton/CUDA measurements from other-int-softmax.md.
    # Ratios are relative to the fused Triton online FP softmax latency.
    2048: {
        "rows": 32768,
        "elements": 67108864,
        "triton_online_fp_ms": 0.107,
        "illm_di_ms": 0.179,
        "intattention_ms": 0.117,
        "illm_di_scale": 1.671,
        "intattention_scale": 1.093,
    },
    4096: {
        "rows": 16384,
        "elements": 67108864,
        "triton_online_fp_ms": 0.108,
        "illm_di_ms": 0.181,
        "intattention_ms": 0.120,
        "illm_di_scale": 1.685,
        "intattention_scale": 1.118,
    },
    8192: {
        "rows": 8192,
        "elements": 67108864,
        "triton_online_fp_ms": 0.111,
        "illm_di_ms": 0.188,
        "intattention_ms": 0.123,
        "illm_di_scale": 1.702,
        "intattention_scale": 1.114,
    },
    16384: {
        "rows": 4096,
        "elements": 67108864,
        "triton_online_fp_ms": 0.120,
        "illm_di_ms": 0.203,
        "intattention_ms": 0.144,
        "illm_di_scale": 1.699,
        "intattention_scale": 1.201,
    },
    32768: {
        "rows": 2048,
        "elements": 67108864,
        "triton_online_fp_ms": 0.130,
        "illm_di_ms": 0.215,
        "intattention_ms": 0.158,
        "illm_di_scale": 1.654,
        "intattention_scale": 1.218,
    },
    65536: {
        "rows": 1024,
        "elements": 67108864,
        "triton_online_fp_ms": 0.154,
        "illm_di_ms": 0.254,
        "intattention_ms": 0.191,
        "illm_di_scale": 1.653,
        "intattention_scale": 1.242,
    },
    131072: {
        "rows": 512,
        "elements": 67108864,
        "triton_online_fp_ms": 0.223,
        "illm_di_ms": 0.313,
        "intattention_ms": 0.283,
        "illm_di_scale": 1.403,
        "intattention_scale": 1.268,
    },
    262144: {
        "rows": 256,
        "elements": 67108864,
        "triton_online_fp_ms": 0.383,
        "illm_di_ms": 0.514,
        "intattention_ms": 0.488,
        "illm_di_scale": 1.342,
        "intattention_scale": 1.273,
    },
    524288: {
        "rows": 128,
        "elements": 67108864,
        "triton_online_fp_ms": 0.708,
        "illm_di_ms": 0.938,
        "intattention_ms": 0.905,
        "illm_di_scale": 1.325,
        "intattention_scale": 1.279,
    },
}

INT_SOFTMAX_SCALE_TABLES = {
    "illm_di": {
        seq_len: row["illm_di_scale"]
        for seq_len, row in INT_SOFTMAX_MICROBENCHMARK.items()
    },
    "intattention": {
        seq_len: row["intattention_scale"]
        for seq_len, row in INT_SOFTMAX_MICROBENCHMARK.items()
    },
}


def _coerce_scale_table(raw_table):
    if raw_table is None:
        return None
    if isinstance(raw_table, (int, float, str)):
        return {"default": float(raw_table)}
    if not isinstance(raw_table, dict):
        raise ValueError(
            "flashattention_softmax scale tables must be a scalar or a dict."
        )
    table = {}
    for key, value in raw_table.items():
        if str(key).lower() == "default":
            table["default"] = float(value)
        else:
            table[int(key)] = float(value)
    return table


def _sequence_scale_from_table(table, seq_len: int) -> float:
    if not table:
        return 1.0
    if seq_len in table:
        return table[seq_len]
    numeric_lengths = sorted(key for key in table if isinstance(key, int))
    if not numeric_lengths:
        return float(table.get("default", 1.0))
    if seq_len <= numeric_lengths[0]:
        return table[numeric_lengths[0]]
    if seq_len >= numeric_lengths[-1]:
        return table[numeric_lengths[-1]]

    for lower, upper in zip(numeric_lengths, numeric_lengths[1:]):
        if lower <= seq_len <= upper:
            # Lengths are swept on powers of two; interpolate on log2(length).
            weight = (log2(seq_len) - log2(lower)) / (log2(upper) - log2(lower))
            return table[lower] + weight * (table[upper] - table[lower])
    return float(table.get("default", 1.0))


class Attention(Operator):
    variant_name = "attention"
    _SOFTMAX_ACCUM_MIN_WORD_SIZE = data_type_dict["fp16"].word_size

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.batch_size = None
        self.num_heads = None
        self.q_len = None
        self.kv_len = None
        self.head_dim = None
        self.output_shape = None
        self.q_mul_k_latency = 0
        self.softmax_latency = 0
        self.a_mul_v_latency = 0

    def __call__(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert self.data_type == query.data_type == key.data_type == value.data_type
        assert len(query.shape) == 4
        assert len(key.shape) == 4
        assert len(value.shape) == 4
        assert query.shape[0] == key.shape[0] == value.shape[0]
        assert query.shape[1] == key.shape[1] == value.shape[1]
        assert key.shape[2] == query.shape[3]
        assert key.shape[3] == value.shape[2]
        assert value.shape[3] == query.shape[3]

        self.batch_size = query.shape[0]
        self.num_heads = query.shape[1]
        self.q_len = query.shape[2]
        self.kv_len = key.shape[3]
        self.head_dim = query.shape[3]

        scores = self.Q_mul_K(query, key)
        assert scores.shape == [
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
        ]
        probs = self.A_softmax(scores)
        output = self.A_mul_V(probs, value)
        assert output.shape == [
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.head_dim,
        ]
        self.output_shape = output.shape
        return output

    def combine_latency(
        self,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
    ) -> float:
        raise NotImplementedError

    def _measure_component_latencies(
        self,
        pcb_module: Device,
        mode: str,
        compile_mode: str = None,
    ):
        if mode == "roofline":
            q_mul_k_latency = (
                self.Q_mul_K.roofline_model(pcb_module)
                + pcb_module.compute_module.overhead.matmul
            )
            softmax_latency = (
                self.A_softmax.roofline_model(pcb_module)
                + pcb_module.compute_module.overhead.softmax
            )
            a_mul_v_latency = (
                self.A_mul_V.roofline_model(pcb_module)
                + pcb_module.compute_module.overhead.matmul
            )
        elif mode == "compile":
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(pcb_module, compile_mode)
                + pcb_module.compute_module.overhead.matmul
            )
            softmax_latency = (
                self.A_softmax.compile_and_simulate(pcb_module, compile_mode)
                + pcb_module.compute_module.overhead.softmax
            )
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(pcb_module, compile_mode)
                + pcb_module.compute_module.overhead.matmul
            )
        elif mode == "gpu":
            q_mul_k_latency = self.Q_mul_K.run_on_gpu()
            softmax_latency = self.A_softmax.run_on_gpu()
            a_mul_v_latency = self.A_mul_V.run_on_gpu()
        else:
            raise ValueError(f"Unsupported attention measurement mode '{mode}'.")
        return q_mul_k_latency, softmax_latency, a_mul_v_latency

    def _record_latency(
        self,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
        mode: str,
    ) -> float:
        self.q_mul_k_latency = q_mul_k_latency
        self.softmax_latency = softmax_latency
        self.a_mul_v_latency = a_mul_v_latency
        total_latency = self.combine_latency(
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
        )
        if mode == "roofline":
            self.roofline_latency = total_latency
        elif mode == "compile":
            self.latency = total_latency
        elif mode == "gpu":
            self.latency_on_gpu = total_latency
        return total_latency

    def roofline_model(self, pcb_module: Device):
        return self._record_latency(
            *self._measure_component_latencies(pcb_module, "roofline"),
            mode="roofline",
        )

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        return self._record_latency(
            *self._measure_component_latencies(pcb_module, "compile", compile_mode),
            mode="compile",
        )

    def run_on_gpu(self):
        return self._record_latency(
            *self._measure_component_latencies(None, "gpu"),
            mode="gpu",
        )

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean-like value, got '{raw_value}'.")

    def _int8_match_fp16_tiles_enabled(self) -> bool:
        return self.data_type.name == "int8" and self._env_flag(
            "LLMCOMPASS_INT8_MATCH_FP16_TILES",
            False,
        )

    def _int8_softmax_conversion_enabled(self) -> bool:
        return self.data_type.name == "int8" and self._env_flag(
            "LLMCOMPASS_INT8_SOFTMAX_CONVERSION",
            False,
        )

    def _int8_softmax_config_signature(self):
        return (
            self._int8_match_fp16_tiles_enabled(),
            self._int8_softmax_conversion_enabled(),
        )

    def _softmax_word_size(self) -> int:
        if self._int8_match_fp16_tiles_enabled():
            return max(self.data_type.word_size, self._SOFTMAX_ACCUM_MIN_WORD_SIZE)
        return self.data_type.word_size

    def _softmax_conversion_flops(self, q_rows: int, kv_rows: int) -> int:
        if not self._int8_softmax_conversion_enabled():
            return 0
        return 4 * q_rows * kv_rows


class TrivialAttention(Attention):
    variant_name = "trivial"

    def combine_latency(
        self,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
    ) -> float:
        return q_mul_k_latency + softmax_latency + a_mul_v_latency


class FlashAttention(Attention):
    variant_name = "flashattention"
    _LOOKUP_TABLE_CACHE = {}
    _MATMUL_CYCLE_CACHE = {}
    _SOFTMAX_CYCLE_CACHE = {}
    _COMPILE_RESULT_CACHE = {}
    _FP16_REFERENCE_MAPPING_CACHE = {}
    _CACHE_STATS = {
        "lookup_hits": 0,
        "lookup_misses": 0,
        "matmul_cycle_hits": 0,
        "matmul_cycle_misses": 0,
        "softmax_cycle_hits": 0,
        "softmax_cycle_misses": 0,
        "compile_hits": 0,
        "compile_misses": 0,
    }

    class Mapping:
        def __init__(
            self,
            q_tile_size: int,
            kv_tile_size: int,
            is_double_buffering: bool,
            workspace_bytes: int,
        ) -> None:
            self.q_tile_size = q_tile_size
            self.kv_tile_size = kv_tile_size
            self.is_double_buffering = is_double_buffering
            self.workspace_bytes = workspace_bytes

    def __init__(self, data_type: DataType):
        super().__init__(data_type)
        self.best_mapping = None
        self.look_up_table = None
        self.hbm_io_latency = 0
        self.onchip_io_latency = 0
        self.compute_latency = 0
        self.kernel_overhead = 0
        self._matmul_cycle_cache = {}
        self._softmax_cycle_cache = {}

    @classmethod
    def reset_cache_stats(cls):
        cls._CACHE_STATS = {
            "lookup_hits": 0,
            "lookup_misses": 0,
            "matmul_cycle_hits": 0,
            "matmul_cycle_misses": 0,
            "softmax_cycle_hits": 0,
            "softmax_cycle_misses": 0,
            "compile_hits": 0,
            "compile_misses": 0,
        }

    @classmethod
    def cache_stats(cls):
        stats = dict(cls._CACHE_STATS)
        for prefix in ["lookup", "matmul_cycle", "softmax_cycle", "compile"]:
            hits = stats[f"{prefix}_hits"]
            misses = stats[f"{prefix}_misses"]
            total = hits + misses
            stats[f"{prefix}_hit_ratio"] = hits / total if total else 0.0
        return stats

    @staticmethod
    def _llmcompass_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def combine_latency(
        self,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
    ) -> float:
        return max(q_mul_k_latency + a_mul_v_latency, softmax_latency)

    @staticmethod
    def _candidate_tile_sizes(limit: int):
        if limit <= 1:
            return [1]
        candidates = {limit}
        base = 1 if limit <= 16 else 16
        tile = base
        while tile < limit:
            candidates.add(tile)
            tile *= 2
        return sorted(candidates)

    def _stats_word_size(self, pcb_module: Device) -> int:
        return max(
            self._softmax_word_size(),
            pcb_module.compute_module.core.vector_unit.word_size,
        )

    def _workspace_bytes(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
        is_double_buffering: bool,
    ) -> int:
        word_size = self._softmax_word_size()
        stats_word_size = self._stats_word_size(pcb_module)
        q_bytes = q_tile_size * self.head_dim * word_size
        o_bytes = q_tile_size * self.head_dim * word_size
        score_bytes = q_tile_size * kv_tile_size * word_size
        stats_bytes = q_tile_size * stats_word_size * 2
        kv_tile_bytes = kv_tile_size * self.head_dim * word_size
        kv_stage_bytes = (4 if is_double_buffering else 2) * kv_tile_bytes
        return q_bytes + o_bytes + score_bytes + stats_bytes + kv_stage_bytes

    def _load_look_up_table(self, pcb_module: Device):
        if self.look_up_table is not None:
            return self.look_up_table

        array_height = pcb_module.compute_module.core.systolic_array.array_height
        array_width = pcb_module.compute_module.core.systolic_array.array_width
        cache_key = (array_height, array_width)
        cached = self._LOOKUP_TABLE_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["lookup_hits"] += 1
            self.look_up_table = cached
            return self.look_up_table
        self._CACHE_STATS["lookup_misses"] += 1
        table = Matmul._get_lookup_table(array_height, array_width)
        self._LOOKUP_TABLE_CACHE[cache_key] = table
        self.look_up_table = table
        return self.look_up_table

    def _single_core_matmul_cycles(
        self,
        M: int,
        N: int,
        K: int,
        pcb_module: Device,
    ) -> int:
        if M == 0 or N == 0 or K == 0:
            return 0

        systolic_array = pcb_module.compute_module.core.systolic_array
        vector_unit = pcb_module.compute_module.core.vector_unit
        look_up_table = self._load_look_up_table(pcb_module)
        cache_key = (
            M,
            N,
            K,
            systolic_array.array_height,
            systolic_array.array_width,
            systolic_array.mac_per_cycle_for(self.data_type),
            pcb_module.compute_module.core.systolic_array_count,
            vector_unit.total_vector_flops_per_cycle,
        )
        cached = self._MATMUL_CYCLE_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["matmul_cycle_hits"] += 1
            return cached
        self._CACHE_STATS["matmul_cycle_misses"] += 1

        if M == 1 or N == 1:
            cycles = ceil(2 * M * N * K / vector_unit.total_vector_flops_per_cycle)
            self._MATMUL_CYCLE_CACHE[cache_key] = cycles
            return cycles

        best_cycles = float("inf")
        last_error = None
        for (
            m_tiling_factor,
            n_tiling_factor,
            k_tiling_factor,
        ) in Matmul.find_permutations(
            pcb_module.compute_module.core.systolic_array_count
        ):
            mapped_M = ceil(M / m_tiling_factor)
            mapped_N = ceil(N / n_tiling_factor)
            mapped_K = ceil(K / k_tiling_factor)
            original_cwd = os.getcwd()
            try:
                os.chdir(self._llmcompass_root())
                try:
                    systolic_cycles = Matmul.simulate_systolic_array_cycle_count(
                        look_up_table,
                        mapped_M,
                        mapped_N,
                        mapped_K,
                        systolic_array.array_height,
                        systolic_array.array_width,
                        systolic_array.mac_per_cycle_for(self.data_type),
                        "os",
                        simulation_mode="lookup",
                    )
                except Exception as exc:
                    last_error = RuntimeError(
                        "FlashAttention tile cycle lookup failed for "
                        f"M={mapped_M}, N={mapped_N}, K={mapped_K}, "
                        f"array={systolic_array.array_height}x{systolic_array.array_width}. "
                        "The fused kernel model is configured to fail fast instead of "
                        "using an analytical fallback."
                    )
                    continue
                tile_cycles = ceil(
                    systolic_cycles
                    + (k_tiling_factor - 1)
                    * M
                    * N
                    / vector_unit.total_vector_flops_per_cycle
                )
            finally:
                os.chdir(original_cwd)
            best_cycles = min(best_cycles, tile_cycles)

        if best_cycles == float("inf"):
            raise last_error or RuntimeError(
                "FlashAttention tile cycle lookup failed for every "
                f"systolic-array permutation at M={M}, N={N}, K={K}."
            )
        best_cycles = int(best_cycles)
        self._MATMUL_CYCLE_CACHE[cache_key] = best_cycles
        return best_cycles

    def _online_softmax_cycles(
        self,
        q_rows: int,
        kv_rows: int,
        pcb_module: Device,
    ) -> int:
        if q_rows == 0 or kv_rows == 0:
            return 0

        vector_unit = pcb_module.compute_module.core.vector_unit
        cache_key = (
            self.data_type.name,
            self._int8_softmax_config_signature(),
            q_rows,
            kv_rows,
            self.head_dim,
            vector_unit.total_vector_flops_per_cycle,
            vector_unit.total_mufu_ops_per_cycle,
        )
        cached = self._SOFTMAX_CYCLE_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["softmax_cycle_hits"] += 1
            return cached
        self._CACHE_STATS["softmax_cycle_misses"] += 1

        score_elements = q_rows * kv_rows
        score_flops = score_elements * 6
        conversion_flops = self._softmax_conversion_flops(q_rows, kv_rows)
        state_update_flops = q_rows * 8
        output_rescale_flops = q_rows * self.head_dim * 3
        cycles = ceil(
            vector_unit.vector_cycles(
                score_flops
                + conversion_flops
                + state_update_flops
                + output_rescale_flops
            )
            + vector_unit.mufu_cycles(score_elements)
        )
        self._SOFTMAX_CYCLE_CACHE[cache_key] = cycles
        return cycles

    def _per_head_stage_cycles(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
    ):
        q_full_tiles = self.q_len // q_tile_size
        q_remainder = self.q_len % q_tile_size
        kv_full_tiles = self.kv_len // kv_tile_size
        kv_remainder = self.kv_len % kv_tile_size

        def sum_tile_cycles(tile_cycle_fn):
            total_cycles = 0
            if q_full_tiles and kv_full_tiles:
                total_cycles += (
                    q_full_tiles
                    * kv_full_tiles
                    * tile_cycle_fn(q_tile_size, kv_tile_size)
                )
            if q_full_tiles and kv_remainder:
                total_cycles += q_full_tiles * tile_cycle_fn(
                    q_tile_size,
                    kv_remainder,
                )
            if q_remainder and kv_full_tiles:
                total_cycles += kv_full_tiles * tile_cycle_fn(
                    q_remainder,
                    kv_tile_size,
                )
            if q_remainder and kv_remainder:
                total_cycles += tile_cycle_fn(q_remainder, kv_remainder)
            return total_cycles

        q_mul_k_cycles = sum_tile_cycles(
            lambda q_rows, kv_rows: self._single_core_matmul_cycles(
                q_rows,
                kv_rows,
                self.head_dim,
                pcb_module,
            )
        )
        a_mul_v_cycles = sum_tile_cycles(
            lambda q_rows, kv_rows: self._single_core_matmul_cycles(
                q_rows,
                self.head_dim,
                kv_rows,
                pcb_module,
            )
        )
        softmax_cycles = sum_tile_cycles(
            lambda q_rows, kv_rows: self._online_softmax_cycles(
                q_rows,
                kv_rows,
                pcb_module,
            )
        )
        return q_mul_k_cycles, softmax_cycles, a_mul_v_cycles

    def _fused_hbm_bytes(self, q_tile_size: int) -> int:
        heads_total = self.batch_size * self.num_heads
        q_tile_count_per_head = ceil(self.q_len / q_tile_size)
        q_and_output_bytes = heads_total * self.q_len * self.head_dim * self.data_type.word_size * 2
        kv_stream_bytes = (
            heads_total
            * q_tile_count_per_head
            * self.kv_len
            * self.head_dim
            * self.data_type.word_size
            * 2
        )
        return q_and_output_bytes + kv_stream_bytes

    def _fused_onchip_bytes(self, q_tile_size: int) -> int:
        return self._fused_hbm_bytes(q_tile_size)

    def _kernel_overhead(self, pcb_module: Device) -> float:
        overhead = pcb_module.compute_module.overhead
        return max(overhead.matmul, overhead.softmax)

    def _compile_cache_key(self, pcb_module: Device, compile_mode: str):
        compute_module = pcb_module.compute_module
        core = compute_module.core
        systolic_array = core.systolic_array
        vector_unit = core.vector_unit
        return (
            self.data_type.name,
            self._int8_softmax_config_signature(),
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            compile_mode,
            systolic_array.array_height,
            systolic_array.array_width,
            systolic_array.mac_per_cycle_for(self.data_type),
            core.systolic_array_count,
            compute_module.core_count,
            core.SRAM_size,
            compute_module.l2_size,
            compute_module.l2_bandwidth_per_cycle,
            compute_module.clock_freq,
            vector_unit.total_vector_flops_per_cycle,
            vector_unit.total_mufu_ops_per_cycle,
            vector_unit.flops_per_exp,
            pcb_module.io_module.bandwidth,
        )

    def _fp16_reference_mapping_cache_key(
        self,
        pcb_module: Device,
        compile_mode: str,
    ):
        compute_module = pcb_module.compute_module
        core = compute_module.core
        systolic_array = core.systolic_array
        vector_unit = core.vector_unit
        return (
            self._int8_softmax_config_signature(),
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            compile_mode,
            systolic_array.array_height,
            systolic_array.array_width,
            systolic_array.mac_per_cycle_for(data_type_dict["fp16"]),
            core.systolic_array_count,
            compute_module.core_count,
            core.SRAM_size,
            compute_module.l2_size,
            compute_module.l2_bandwidth_per_cycle,
            compute_module.clock_freq,
            vector_unit.total_vector_flops_per_cycle,
            vector_unit.total_mufu_ops_per_cycle,
            vector_unit.flops_per_exp,
            pcb_module.io_module.bandwidth,
        )

    def _fp16_reference_pcb_module(self, pcb_module: Device) -> Device:
        return deepcopy(pcb_module)

    def _fp16_reference_mapping(self, pcb_module: Device, compile_mode: str):
        cache_key = self._fp16_reference_mapping_cache_key(pcb_module, compile_mode)
        cached = self._FP16_REFERENCE_MAPPING_CACHE.get(cache_key)
        if cached is not None:
            return cached
        reference = FlashAttention(data_type_dict["fp16"])
        query = Tensor(
            [self.batch_size, self.num_heads, self.q_len, self.head_dim],
            data_type_dict["fp16"],
        )
        key = Tensor(
            [self.batch_size, self.num_heads, self.head_dim, self.kv_len],
            data_type_dict["fp16"],
        )
        value = Tensor(
            [self.batch_size, self.num_heads, self.kv_len, self.head_dim],
            data_type_dict["fp16"],
        )
        reference(query, key, value)
        _, mapping, *_ = reference._search_best_mapping(
            self._fp16_reference_pcb_module(pcb_module)
        )
        self._FP16_REFERENCE_MAPPING_CACHE[cache_key] = mapping
        return mapping

    def _estimate_mapping_latency(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
        is_double_buffering: bool,
    ):
        heads_total = self.batch_size * self.num_heads
        total_q_tiles = heads_total * ceil(self.q_len / q_tile_size)
        effective_parallelism = max(
            1,
            min(pcb_module.compute_module.core_count, total_q_tiles),
        )
        clock_freq = pcb_module.compute_module.clock_freq
        q_mul_k_cycles_per_head, softmax_cycles_per_head, a_mul_v_cycles_per_head = (
            self._per_head_stage_cycles(q_tile_size, kv_tile_size, pcb_module)
        )
        q_mul_k_latency = (
            heads_total * q_mul_k_cycles_per_head / effective_parallelism / clock_freq
        )
        softmax_latency = (
            heads_total * softmax_cycles_per_head / effective_parallelism / clock_freq
        )
        a_mul_v_latency = (
            heads_total * a_mul_v_cycles_per_head / effective_parallelism / clock_freq
        )
        compute_latency = self.combine_latency(
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
        )

        hbm_bytes = self._fused_hbm_bytes(q_tile_size)
        if pcb_module.io_module.bandwidth == float("inf"):
            hbm_latency = 0
        else:
            hbm_latency = hbm_bytes / pcb_module.io_module.bandwidth
        onchip_latency = self._fused_onchip_bytes(q_tile_size) / (
            pcb_module.compute_module.l2_bandwidth_per_cycle * clock_freq
        )

        kernel_overhead = self._kernel_overhead(pcb_module)
        total_latency = max(compute_latency, hbm_latency, onchip_latency) + kernel_overhead
        mapping = self.Mapping(
            q_tile_size,
            kv_tile_size,
            is_double_buffering,
            self._workspace_bytes(
                q_tile_size,
                kv_tile_size,
                pcb_module,
                is_double_buffering,
            ),
        )
        return (
            total_latency,
            mapping,
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
            compute_latency,
            hbm_latency,
            onchip_latency,
            kernel_overhead,
        )

    def _search_best_mapping(self, pcb_module: Device):
        best_result = None
        sram_size = pcb_module.compute_module.core.SRAM_size
        l2_size = pcb_module.compute_module.l2_size
        for q_tile_size in self._candidate_tile_sizes(self.q_len):
            for kv_tile_size in self._candidate_tile_sizes(self.kv_len):
                for is_double_buffering in [True, False]:
                    workspace_bytes = self._workspace_bytes(
                        q_tile_size,
                        kv_tile_size,
                        pcb_module,
                        is_double_buffering,
                    )
                    if workspace_bytes > sram_size:
                        continue
                    if workspace_bytes > l2_size:
                        continue
                    try:
                        result = self._estimate_mapping_latency(
                            q_tile_size,
                            kv_tile_size,
                            pcb_module,
                            is_double_buffering,
                        )
                    except RuntimeError:
                        continue
                    if best_result is None or result[0] < best_result[0]:
                        best_result = result
        if best_result is None:
            raise ValueError(
                "No feasible FlashAttention tile mapping fits the modeled SRAM."
            )
        return best_result

    def _record_fused_result(
        self,
        total_latency: float,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
        compute_latency: float,
        hbm_io_latency: float,
        onchip_io_latency: float,
        kernel_overhead: float,
        mapping: "FlashAttention.Mapping" = None,
        mode: str = "compile",
    ) -> float:
        self.q_mul_k_latency = q_mul_k_latency
        self.softmax_latency = softmax_latency
        self.a_mul_v_latency = a_mul_v_latency
        self.compute_latency = compute_latency
        self.hbm_io_latency = hbm_io_latency
        self.onchip_io_latency = onchip_io_latency
        self.kernel_overhead = kernel_overhead
        self.best_mapping = mapping
        if mode == "roofline":
            self.roofline_latency = total_latency
        elif mode == "compile":
            self.latency = total_latency
        elif mode == "gpu":
            self.latency_on_gpu = total_latency
        return total_latency

    def roofline_model(self, pcb_module: Device):
        heads_total = self.batch_size * self.num_heads
        word_size = self.data_type.word_size
        total_qk_flops = (
            2 * heads_total * self.q_len * self.kv_len * self.head_dim
        )
        total_av_flops = total_qk_flops
        vector_unit = pcb_module.compute_module.core.vector_unit
        total_softmax_vector_flops = (
            heads_total * self.q_len * self.kv_len * 6
            + heads_total * self._softmax_conversion_flops(self.q_len, self.kv_len)
            + heads_total * self.q_len * (8 + 3 * self.head_dim)
        )
        total_softmax_mufu_ops = heads_total * self.q_len * self.kv_len
        systolic_flops = pcb_module.compute_module.total_systolic_array_flops_for(
            self.data_type
        )
        q_mul_k_latency = total_qk_flops / systolic_flops
        a_mul_v_latency = total_av_flops / systolic_flops
        softmax_latency = (
            pcb_module.compute_module.vector_latency(total_softmax_vector_flops)
            + pcb_module.compute_module.mufu_latency(total_softmax_mufu_ops)
        )
        compute_latency = self.combine_latency(
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
        )
        q_and_output_bytes = heads_total * self.q_len * self.head_dim * word_size * 2
        kv_stream_bytes = heads_total * self.q_len * self.kv_len * self.head_dim * word_size * 2
        hbm_io_latency = 0
        if pcb_module.io_module.bandwidth != float("inf"):
            hbm_io_latency = (
                q_and_output_bytes + kv_stream_bytes
            ) / pcb_module.io_module.bandwidth
        onchip_io_latency = (
            q_and_output_bytes + kv_stream_bytes
        ) / (
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq
        )
        kernel_overhead = self._kernel_overhead(pcb_module)
        total_latency = max(compute_latency, hbm_io_latency, onchip_io_latency) + kernel_overhead
        return self._record_fused_result(
            total_latency,
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
            compute_latency,
            hbm_io_latency,
            onchip_io_latency,
            kernel_overhead,
            mapping=None,
            mode="roofline",
        )

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        cache_key = self._compile_cache_key(pcb_module, compile_mode)
        cached = self._COMPILE_RESULT_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["compile_hits"] += 1
            return self._record_fused_result(
                *cached,
                mode="compile",
            )
        self._CACHE_STATS["compile_misses"] += 1
        if self._int8_match_fp16_tiles_enabled():
            reference_mapping = self._fp16_reference_mapping(pcb_module, compile_mode)
            (
                total_latency,
                mapping,
                q_mul_k_latency,
                softmax_latency,
                a_mul_v_latency,
                compute_latency,
                hbm_io_latency,
                onchip_io_latency,
                kernel_overhead,
            ) = self._estimate_mapping_latency(
                reference_mapping.q_tile_size,
                reference_mapping.kv_tile_size,
                pcb_module,
                reference_mapping.is_double_buffering,
            )
        else:
            (
                total_latency,
                mapping,
                q_mul_k_latency,
                softmax_latency,
                a_mul_v_latency,
                compute_latency,
                hbm_io_latency,
                onchip_io_latency,
                kernel_overhead,
            ) = self._search_best_mapping(pcb_module)
        self._COMPILE_RESULT_CACHE[cache_key] = (
            total_latency,
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
            compute_latency,
            hbm_io_latency,
            onchip_io_latency,
            kernel_overhead,
            mapping,
        )
        return self._record_fused_result(
            total_latency,
            q_mul_k_latency,
            softmax_latency,
            a_mul_v_latency,
            compute_latency,
            hbm_io_latency,
            onchip_io_latency,
            kernel_overhead,
            mapping=mapping,
            mode="compile",
        )


class FlashAttentionScaledSoftmax(FlashAttention):
    variant_name = "flashattention_scaled_softmax"
    softmax_scale_key = None
    softmax_scale_label = "scaled"
    _SOFTMAX_CYCLE_CACHE = {}
    _COMPILE_RESULT_CACHE = {}

    def _configured_softmax_scale_table(self, pcb_module: Device):
        default_table = INT_SOFTMAX_SCALE_TABLES.get(self.softmax_scale_key, {})
        raw_scales = getattr(pcb_module, "operator_latency_scales", {})
        raw_table = None
        if isinstance(raw_scales, dict):
            flashattention_softmax = raw_scales.get("flashattention_softmax", {})
            if isinstance(flashattention_softmax, dict):
                raw_table = flashattention_softmax.get(self.softmax_scale_key)
        return _coerce_scale_table(raw_table) or default_table

    def _softmax_scale_for_sequence(self, pcb_module: Device) -> float:
        return _sequence_scale_from_table(
            self._configured_softmax_scale_table(pcb_module),
            int(self.kv_len),
        )

    def _softmax_scale_signature(self, pcb_module: Device):
        table = self._configured_softmax_scale_table(pcb_module)
        if isinstance(table, dict):
            return tuple(sorted(table.items(), key=lambda item: str(item[0])))
        return table

    def _online_softmax_cycles(
        self,
        q_rows: int,
        kv_rows: int,
        pcb_module: Device,
    ) -> int:
        base_cycles = super()._online_softmax_cycles(q_rows, kv_rows, pcb_module)
        return ceil(base_cycles * self._softmax_scale_for_sequence(pcb_module))

    def _compile_cache_key(self, pcb_module: Device, compile_mode: str):
        return (
            super()._compile_cache_key(pcb_module, compile_mode),
            self.softmax_scale_key,
            self._softmax_scale_signature(pcb_module),
        )

    def _fp16_reference_mapping_cache_key(
        self,
        pcb_module: Device,
        compile_mode: str,
    ):
        return (
            super()._fp16_reference_mapping_cache_key(pcb_module, compile_mode),
            self.softmax_scale_key,
            self._softmax_scale_signature(pcb_module),
        )


class FlashAttentionILLM(FlashAttentionScaledSoftmax):
    variant_name = "flashattention_illm"
    softmax_scale_key = "illm_di"
    softmax_scale_label = "I-LLM DI-Softmax"
    _SOFTMAX_CYCLE_CACHE = {}
    _COMPILE_RESULT_CACHE = {}
    _FP16_REFERENCE_MAPPING_CACHE = {}


class FlashAttentionIntAttention(FlashAttentionScaledSoftmax):
    variant_name = "flashattention_intattention"
    softmax_scale_key = "intattention"
    softmax_scale_label = "IntAttention IndexSoftmax"
    _SOFTMAX_CYCLE_CACHE = {}
    _COMPILE_RESULT_CACHE = {}
    _FP16_REFERENCE_MAPPING_CACHE = {}


class FlashAttentionCustomSA(Attention):
    variant_name = "flashattention_customsa"
    _DEFAULT_STAGE_OVERHEAD_CYCLES = 8
    _COMPILE_RESULT_CACHE = {}
    _REFERENCE_RESULT_CACHE = {}
    _CACHE_STATS = {
        "compile_hits": 0,
        "compile_misses": 0,
        "reference_hits": 0,
        "reference_misses": 0,
    }

    class Mapping:
        def __init__(
            self,
            primitive_tile_size: int,
            logical_q_tile_size: int,
            q_tile_size: int,
            kv_tile_size: int,
            workspace_bytes: int,
            hbm_bytes: int,
        ) -> None:
            self.primitive_tile_size = primitive_tile_size
            self.logical_q_tile_size = logical_q_tile_size
            self.q_tile_size = q_tile_size
            self.kv_tile_size = kv_tile_size
            self.workspace_bytes = workspace_bytes
            self.hbm_bytes = hbm_bytes

    def __init__(self, data_type: DataType):
        super().__init__(data_type)
        self.best_mapping = None
        self.fused_core_cycles = 0
        self.fused_core_latency = 0
        self.hbm_bytes = 0
        self.hbm_io_latency = 0
        self.onchip_io_latency = 0
        self.kernel_overhead = 0
        self.steady_cycles_per_tile = 0
        self.drain_cycles_per_tile = 0

    @classmethod
    def reset_cache_stats(cls):
        cls._CACHE_STATS = {
            "compile_hits": 0,
            "compile_misses": 0,
            "reference_hits": 0,
            "reference_misses": 0,
        }

    @classmethod
    def cache_stats(cls):
        stats = dict(cls._CACHE_STATS)
        for prefix in ["compile", "reference"]:
            total = stats[f"{prefix}_hits"] + stats[f"{prefix}_misses"]
            stats[f"{prefix}_hit_ratio"] = (
                stats[f"{prefix}_hits"] / total if total else 0.0
            )
        return stats

    def combine_latency(
        self,
        q_mul_k_latency: float,
        softmax_latency: float,
        a_mul_v_latency: float,
    ) -> float:
        return q_mul_k_latency + softmax_latency + a_mul_v_latency

    def _primitive_tile_size(self, pcb_module: Device) -> int:
        systolic_array = pcb_module.compute_module.core.systolic_array
        if systolic_array.array_height != systolic_array.array_width:
            raise ValueError(
                "FlashAttentionCustomSA requires a square systolic array, "
                f"got {systolic_array.array_height}x{systolic_array.array_width}."
            )
        if systolic_array.array_height <= 0:
            raise ValueError("Systolic array size must be positive.")
        return systolic_array.array_height

    def _customsa_tile_search_enabled(self) -> bool:
        return self._env_flag(
            "LLMCOMPASS_CUSTOMSA_SEARCH_TILES",
            False,
        )

    def _software_tile_candidates(self, limit: int, primitive_tile_size: int):
        if limit <= primitive_tile_size:
            return [limit]
        candidates = {limit}
        tile_size = primitive_tile_size
        while tile_size < limit:
            candidates.add(tile_size)
            tile_size *= 2
        return sorted(candidates)

    def _query_pack_factor(self) -> int:
        if self.data_type.name == "int8":
            return 2
        return 1

    def _stage_overhead_cycles(self) -> int:
        raw_value = os.environ.get(
            "LLMCOMPASS_CUSTOMSA_STAGE_OVERHEAD_CYCLES",
            str(self._DEFAULT_STAGE_OVERHEAD_CYCLES),
        ).strip()
        stage_overhead_cycles = int(raw_value)
        if stage_overhead_cycles < 0:
            raise ValueError(
                "LLMCOMPASS_CUSTOMSA_STAGE_OVERHEAD_CYCLES must be non-negative."
            )
        return stage_overhead_cycles

    def _stats_word_size(self, pcb_module: Device) -> int:
        return max(
            self._softmax_word_size(),
            pcb_module.compute_module.core.vector_unit.word_size,
        )

    def _workspace_bytes(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
    ) -> int:
        word_size = self._softmax_word_size()
        stats_word_size = self._stats_word_size(pcb_module)
        q_bytes = q_tile_size * self.head_dim * word_size
        kv_bytes = kv_tile_size * self.head_dim * word_size
        o_bytes = q_tile_size * self.head_dim * word_size
        stats_bytes = q_tile_size * 2 * stats_word_size
        return q_bytes + 2 * kv_bytes + o_bytes + stats_bytes

    def _fused_hbm_bytes(self, q_tile_size: int) -> int:
        heads_total = self.batch_size * self.num_heads
        q_tile_count_per_head = ceil(self.q_len / q_tile_size)
        q_and_output_bytes = (
            heads_total
            * self.q_len
            * self.head_dim
            * self.data_type.word_size
            * 2
        )
        kv_stream_bytes = (
            heads_total
            * q_tile_count_per_head
            * self.kv_len
            * self.head_dim
            * self.data_type.word_size
            * 2
        )
        return q_and_output_bytes + kv_stream_bytes

    def _fused_onchip_bytes(self, q_tile_size: int) -> int:
        return self._fused_hbm_bytes(q_tile_size)

    def _fused_core_cycles(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
    ):
        primitive_tile_size = self._primitive_tile_size(pcb_module)
        logical_q_tile_size = q_tile_size
        total_q_tiles = (
            self.batch_size * self.num_heads * ceil(self.q_len / q_tile_size)
        )
        kv_tiles_per_q = ceil(self.kv_len / kv_tile_size)
        mac_per_cycle = pcb_module.compute_module.core.systolic_array.mac_per_cycle_for(
            self.data_type
        )
        if mac_per_cycle <= 0:
            raise ValueError("CustomSA requires a positive mac_per_cycle.")
        stage_overhead_cycles = self._stage_overhead_cycles()
        conversion_cycles = pcb_module.compute_module.core.vector_unit.vector_cycles(
            self._softmax_conversion_flops(q_tile_size, kv_tile_size)
        )
        steady_cycles_per_tile = ceil(
            2
            * self.head_dim
            * q_tile_size
            * kv_tile_size
            / (mac_per_cycle * (primitive_tile_size**2))
            + q_tile_size * kv_tile_size / (mac_per_cycle * primitive_tile_size)
            + conversion_cycles
            + 2 * primitive_tile_size
            + stage_overhead_cycles
        )
        drain_cycles_per_tile = 0
        cycles_per_q_tile = kv_tiles_per_q * steady_cycles_per_tile
        parallel_units = max(
            1,
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.systolic_array_count,
        )
        total_cycles = ceil(total_q_tiles / parallel_units) * cycles_per_q_tile
        return (
            total_cycles,
            primitive_tile_size,
            logical_q_tile_size,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        )

    def _kernel_overhead(self, pcb_module: Device) -> float:
        overhead = pcb_module.compute_module.overhead
        return max(overhead.matmul, overhead.softmax)

    def _compile_cache_key(self, pcb_module: Device, compile_mode: str):
        compute_module = pcb_module.compute_module
        core = compute_module.core
        systolic_array = core.systolic_array
        vector_unit = core.vector_unit
        return (
            self.data_type.name,
            self._int8_softmax_config_signature(),
            self._customsa_tile_search_enabled(),
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            compile_mode,
            systolic_array.array_height,
            systolic_array.array_width,
            systolic_array.mac_per_cycle_for(self.data_type),
            core.systolic_array_count,
            compute_module.core_count,
            core.SRAM_size,
            compute_module.l2_size,
            compute_module.l2_bandwidth_per_cycle,
            compute_module.clock_freq,
            vector_unit.total_vector_flops_per_cycle,
            vector_unit.total_mufu_ops_per_cycle,
            vector_unit.flops_per_exp,
            pcb_module.io_module.bandwidth,
        )

    def _reference_cache_key(self, pcb_module: Device, compile_mode: str):
        compute_module = pcb_module.compute_module
        core = compute_module.core
        systolic_array = core.systolic_array
        vector_unit = core.vector_unit
        return (
            self.data_type.name,
            self._int8_softmax_config_signature(),
            self.batch_size,
            self.num_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            compile_mode,
            systolic_array.array_height,
            systolic_array.array_width,
            systolic_array.mac_per_cycle_for(self.data_type),
            core.systolic_array_count,
            compute_module.core_count,
            core.SRAM_size,
            compute_module.l2_size,
            compute_module.l2_bandwidth_per_cycle,
            compute_module.clock_freq,
            vector_unit.total_vector_flops_per_cycle,
            vector_unit.total_mufu_ops_per_cycle,
            vector_unit.flops_per_exp,
            pcb_module.io_module.bandwidth,
        )

    def _build_reference_flashattention(self, pcb_module: Device, compile_mode: str):
        cache_key = self._reference_cache_key(pcb_module, compile_mode)
        cached = self._REFERENCE_RESULT_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["reference_hits"] += 1
            return cached
        self._CACHE_STATS["reference_misses"] += 1
        reference = FlashAttention(self.data_type)
        query = Tensor(
            [self.batch_size, self.num_heads, self.q_len, self.head_dim],
            self.data_type,
        )
        key = Tensor(
            [self.batch_size, self.num_heads, self.head_dim, self.kv_len],
            self.data_type,
        )
        value = Tensor(
            [self.batch_size, self.num_heads, self.kv_len, self.head_dim],
            self.data_type,
        )
        reference(query, key, value)
        reference.compile_and_simulate(pcb_module, compile_mode)
        if reference.best_mapping is None:
            raise ValueError(
                "FlashAttentionCustomSA expects the reference FlashAttention path "
                "to produce a best_mapping."
            )
        result = {
            "q_tile_size": reference.best_mapping.q_tile_size,
            "kv_tile_size": reference.best_mapping.kv_tile_size,
            "hbm_io_latency": reference.hbm_io_latency,
            "onchip_io_latency": reference.onchip_io_latency,
            "hbm_bytes": reference._fused_hbm_bytes(reference.best_mapping.q_tile_size),
            "onchip_bytes": reference._fused_onchip_bytes(reference.best_mapping.q_tile_size),
        }
        self._REFERENCE_RESULT_CACHE[cache_key] = result
        return result

    def _estimate_mapping_latency(
        self,
        q_tile_size: int,
        kv_tile_size: int,
        pcb_module: Device,
    ):
        workspace_bytes = self._workspace_bytes(q_tile_size, kv_tile_size, pcb_module)
        if workspace_bytes > pcb_module.compute_module.core.SRAM_size:
            raise ValueError(
                "CustomSA tile mapping does not fit core SRAM workspace constraints."
            )
        if workspace_bytes > pcb_module.compute_module.l2_size:
            raise ValueError(
                "CustomSA tile mapping does not fit modeled L2 workspace constraints."
            )

        (
            fused_core_cycles,
            primitive_tile_size,
            logical_q_tile_size,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        ) = self._fused_core_cycles(q_tile_size, kv_tile_size, pcb_module)
        hbm_bytes = self._fused_hbm_bytes(q_tile_size)
        onchip_bytes = self._fused_onchip_bytes(q_tile_size)
        fused_core_latency = fused_core_cycles / pcb_module.compute_module.clock_freq
        if pcb_module.io_module.bandwidth == float("inf"):
            hbm_io_latency = 0
        else:
            hbm_io_latency = hbm_bytes / pcb_module.io_module.bandwidth
        onchip_io_latency = onchip_bytes / (
            pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq
        )
        kernel_overhead = self._kernel_overhead(pcb_module)
        total_latency = max(
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
        ) + kernel_overhead
        mapping = self.Mapping(
            primitive_tile_size,
            logical_q_tile_size,
            q_tile_size,
            kv_tile_size,
            workspace_bytes,
            hbm_bytes,
        )
        return (
            total_latency,
            mapping,
            fused_core_cycles,
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
            hbm_bytes,
            kernel_overhead,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        )

    def _search_best_mapping(
        self,
        pcb_module: Device,
    ):
        primitive_tile_size = self._primitive_tile_size(pcb_module)
        best_result = None
        for q_tile_size in self._software_tile_candidates(
            self.q_len,
            primitive_tile_size,
        ):
            for kv_tile_size in self._software_tile_candidates(
                self.kv_len,
                primitive_tile_size,
            ):
                try:
                    result = self._estimate_mapping_latency(
                        q_tile_size,
                        kv_tile_size,
                        pcb_module,
                    )
                except ValueError:
                    continue
                if best_result is None or result[0] < best_result[0]:
                    best_result = result
        if best_result is None:
            raise ValueError(
                "No feasible CustomSA tile mapping fits the modeled SRAM/L2 limits."
            )
        return best_result

    def _simulate_with_reference_mapping(
        self,
        pcb_module: Device,
        compile_mode: str,
    ):
        reference = self._build_reference_flashattention(pcb_module, compile_mode)
        q_tile_size = reference["q_tile_size"]
        kv_tile_size = reference["kv_tile_size"]
        (
            fused_core_cycles,
            primitive_tile_size,
            logical_q_tile_size,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        ) = self._fused_core_cycles(q_tile_size, kv_tile_size, pcb_module)
        workspace_bytes = self._workspace_bytes(q_tile_size, kv_tile_size, pcb_module)
        if workspace_bytes > pcb_module.compute_module.core.SRAM_size:
            raise ValueError(
                "Reference FlashAttention mapping does not fit FlashAttentionCustomSA "
                "workspace constraints."
            )

        hbm_bytes = reference["hbm_bytes"]
        onchip_bytes = reference["onchip_bytes"]
        fused_core_latency = fused_core_cycles / pcb_module.compute_module.clock_freq
        hbm_io_latency = reference["hbm_io_latency"]
        onchip_io_latency = reference["onchip_io_latency"]
        kernel_overhead = self._kernel_overhead(pcb_module)
        total_latency = max(
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
        ) + kernel_overhead
        mapping = self.Mapping(
            primitive_tile_size,
            logical_q_tile_size,
            q_tile_size,
            kv_tile_size,
            workspace_bytes,
            hbm_bytes,
        )
        return (
            total_latency,
            mapping,
            fused_core_cycles,
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
            hbm_bytes,
            kernel_overhead,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        )

    def _record_customsa_result(
        self,
        total_latency: float,
        mapping: "FlashAttentionCustomSA.Mapping",
        fused_core_cycles: int,
        fused_core_latency: float,
        hbm_io_latency: float,
        onchip_io_latency: float,
        hbm_bytes: int,
        kernel_overhead: float,
        steady_cycles_per_tile: int,
        drain_cycles_per_tile: int,
        mode: str,
    ) -> float:
        self.best_mapping = mapping
        self.fused_core_cycles = fused_core_cycles
        self.fused_core_latency = fused_core_latency
        self.hbm_bytes = hbm_bytes
        self.hbm_io_latency = hbm_io_latency
        self.onchip_io_latency = onchip_io_latency
        self.kernel_overhead = kernel_overhead
        self.q_mul_k_latency = 0
        self.softmax_latency = 0
        self.a_mul_v_latency = 0
        self.steady_cycles_per_tile = steady_cycles_per_tile
        self.drain_cycles_per_tile = drain_cycles_per_tile
        if mode == "roofline":
            self.roofline_latency = total_latency
        elif mode == "compile":
            self.latency = total_latency
        else:
            raise ValueError(
                f"Unsupported FlashAttentionCustomSA record mode '{mode}'."
            )
        return total_latency

    def _simulate(self, pcb_module: Device, mode: str, compile_mode: str = "heuristic-GPU") -> float:
        if mode != "compile":
            raise ValueError(
            "FlashAttentionCustomSA currently supports compile-mode simulation "
            "only for iso-schedule comparison against FlashAttention."
        )
        cache_key = self._compile_cache_key(pcb_module, compile_mode)
        cached = self._COMPILE_RESULT_CACHE.get(cache_key)
        if cached is not None:
            self._CACHE_STATS["compile_hits"] += 1
            return self._record_customsa_result(
                *cached,
                mode,
            )
        self._CACHE_STATS["compile_misses"] += 1
        if self._customsa_tile_search_enabled():
            (
                total_latency,
                mapping,
                fused_core_cycles,
                fused_core_latency,
                hbm_io_latency,
                onchip_io_latency,
                hbm_bytes,
                kernel_overhead,
                steady_cycles_per_tile,
                drain_cycles_per_tile,
            ) = self._search_best_mapping(pcb_module)
        else:
            (
                total_latency,
                mapping,
                fused_core_cycles,
                fused_core_latency,
                hbm_io_latency,
                onchip_io_latency,
                hbm_bytes,
                kernel_overhead,
                steady_cycles_per_tile,
                drain_cycles_per_tile,
            ) = self._simulate_with_reference_mapping(pcb_module, compile_mode)
        self._COMPILE_RESULT_CACHE[cache_key] = (
            total_latency,
            mapping,
            fused_core_cycles,
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
            hbm_bytes,
            kernel_overhead,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
        )
        return self._record_customsa_result(
            total_latency,
            mapping,
            fused_core_cycles,
            fused_core_latency,
            hbm_io_latency,
            onchip_io_latency,
            hbm_bytes,
            kernel_overhead,
            steady_cycles_per_tile,
            drain_cycles_per_tile,
            mode,
        )

    def roofline_model(self, pcb_module: Device):
        raise NotImplementedError(
            "FlashAttentionCustomSA roofline_model is not implemented for the "
            "iso-schedule comparison path."
        )

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        return self._simulate(pcb_module, "compile", compile_mode)

    def run_on_gpu(self):
        raise NotImplementedError(
            "FlashAttentionCustomSA does not provide a run_on_gpu path. "
            "Use compile_and_simulate with a Device model instead."
        )


def build_attention(
    attention_variant: str,
    data_type: DataType,
) -> Attention:
    if attention_variant == "trivial":
        return TrivialAttention(data_type)
    if attention_variant == "flashattention":
        return FlashAttention(data_type)
    if attention_variant in {"flashattention_illm", "flashattention_illm_di"}:
        return FlashAttentionILLM(data_type)
    if attention_variant == "flashattention_intattention":
        return FlashAttentionIntAttention(data_type)
    if attention_variant == "flashattention_customsa":
        return FlashAttentionCustomSA(data_type)
    raise ValueError(f"Unsupported attention variant '{attention_variant}'.")


class _TensorParallelAttentionBase(Operator):
    def __init__(
        self,
        d_model,
        n_heads,
        device_count,
        data_type: DataType,
        attention_variant: str = "trivial",
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.attention_variant = attention_variant

        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.attention = build_attention(self.attention_variant, data_type)
        self.Q_mul_K = self.attention.Q_mul_K
        self.A_softmax = self.attention.A_softmax
        self.A_mul_V = self.attention.A_mul_V
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)

        self.qkv_latency = 0
        self.q_mul_k_latency = 0
        self.softmax_latency = 0
        self.a_mul_v_latency = 0
        self.h_matmul0_latency = 0
        self.output_shape = None
        self.memory_requirement = 0

    def _project_qkv(self, x: Tensor):
        b, s, d = x.shape
        assert d == self.d_model
        h_per_device = self.n_heads // self.device_count
        d_h = d // self.n_heads

        q = self.Q_proj(x, self.Wq)
        assert q.shape == [b, s, d // self.device_count]
        k = self.K_proj(x, self.Wk)
        assert k.shape == [b, s, d // self.device_count]
        v = self.V_proj(x, self.Wv)
        assert v.shape == [b, s, d // self.device_count]

        q = self.Q_reshape(q, [b, s, h_per_device, d_h])
        k = self.K_reshape(k, [b, s, h_per_device, d_h])
        v = self.V_reshape(v, [b, s, h_per_device, d_h])

        q_t = self.Q_transpose(q, [0, 2, 1, 3])
        assert q_t.shape == [b, h_per_device, s, d_h]
        k_t = self.K_transpose(k, [0, 2, 3, 1])
        assert k_t.shape == [b, h_per_device, d_h, s]
        v_t = self.V_transpose(v, [0, 2, 1, 3])
        assert v_t.shape == [b, h_per_device, s, d_h]
        return q_t, k_t, v_t, b, s, d_h, h_per_device

    def _merge_heads(self, hidden: Tensor, batch_size: int, seq_len: int) -> Tensor:
        hidden = self.H_transpose(hidden, [0, 2, 1, 3])
        assert hidden.shape == [
            batch_size,
            seq_len,
            self.n_heads // self.device_count,
            self.d_model // self.n_heads,
        ]
        hidden = self.H_reshape(
            hidden,
            [batch_size, seq_len, self.d_model // self.device_count],
        )
        assert hidden.shape == [batch_size, seq_len, self.d_model // self.device_count]
        output = self.H_matmul0(hidden, self.W0)
        assert output.shape == [batch_size, seq_len, self.d_model]
        self.output_shape = output.shape
        return output

    def _weight_memory_requirement(self) -> int:
        return (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
        )

    def _matmul_latency(
        self,
        op: Matmul,
        pcb_module: Device,
        mode: str,
        compile_mode: str = None,
    ) -> float:
        if mode == "roofline":
            return (
                op.roofline_model(pcb_module)
                + pcb_module.compute_module.overhead.matmul
            )
        if mode == "compile":
            return (
                op.compile_and_simulate(pcb_module, compile_mode)
                + pcb_module.compute_module.overhead.matmul
            )
        if mode == "gpu":
            return op.run_on_gpu()
        raise ValueError(f"Unsupported attention block measurement mode '{mode}'.")

    def _record_latency(
        self,
        qkv_latency: float,
        attention_latency: float,
        h_matmul0_latency: float,
        mode: str,
    ) -> float:
        self.qkv_latency = qkv_latency
        self.q_mul_k_latency = self.attention.q_mul_k_latency
        self.softmax_latency = self.attention.softmax_latency
        self.a_mul_v_latency = self.attention.a_mul_v_latency
        self.h_matmul0_latency = h_matmul0_latency
        total_latency = qkv_latency + attention_latency + h_matmul0_latency
        if mode == "roofline":
            self.roofline_latency = total_latency
        elif mode == "compile":
            self.latency = total_latency
        elif mode == "gpu":
            self.latency_on_gpu = total_latency
        return total_latency

    def _measure_latency(
        self,
        pcb_module: Device,
        mode: str,
        compile_mode: str = None,
    ) -> float:
        qkv_latency = 3 * self._matmul_latency(
            self.Q_proj,
            pcb_module,
            mode,
            compile_mode,
        )
        if mode == "roofline":
            attention_latency = self.attention.roofline_model(pcb_module)
        elif mode == "compile":
            attention_latency = self.attention.compile_and_simulate(
                pcb_module,
                compile_mode,
            )
        elif mode == "gpu":
            attention_latency = self.attention.run_on_gpu()
        else:
            raise ValueError(f"Unsupported attention block measurement mode '{mode}'.")
        h_matmul0_latency = self._matmul_latency(
            self.H_matmul0,
            pcb_module,
            mode,
            compile_mode,
        )
        return self._record_latency(
            qkv_latency,
            attention_latency,
            h_matmul0_latency,
            mode,
        )

    def roofline_model(self, pcb_module: Device):
        return self._measure_latency(pcb_module, "roofline")

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        return self._measure_latency(pcb_module, "compile", compile_mode)

    def run_on_gpu(self):
        return self._measure_latency(None, "gpu")


class MultiHeadAttentionInitComputationTP(_TensorParallelAttentionBase):
    def __call__(self, x: Tensor) -> Tensor:
        q_t, k_t, v_t, batch_size, seq_len, _, _ = self._project_qkv(x)
        hidden = self.attention(q_t, k_t, v_t)
        self.memory_requirement = self._weight_memory_requirement()
        return self._merge_heads(hidden, batch_size, seq_len)


class MultiHeadAttentionAutoRegressionTP(_TensorParallelAttentionBase):
    def __init__(
        self,
        d_model,
        n_heads,
        device_count,
        data_type: DataType,
        attention_variant: str = "trivial",
    ):
        super().__init__(
            d_model,
            n_heads,
            device_count,
            data_type,
            attention_variant=attention_variant,
        )
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.K_cache = None
        self.V_cache = None

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        q_t, k_t, v_t, batch_size, _, d_h, h_per_device = self._project_qkv(x)
        self.K_cache = Tensor([batch_size, h_per_device, d_h, seq_len], self.data_type)
        self.V_cache = Tensor([batch_size, h_per_device, seq_len, d_h], self.data_type)
        k_t = self.K_concat(self.K_cache, k_t, 3)
        assert k_t.shape == [batch_size, h_per_device, d_h, seq_len + 1]
        v_t = self.V_concat(self.V_cache, v_t, 2)
        assert v_t.shape == [batch_size, h_per_device, seq_len + 1, d_h]
        hidden = self.attention(q_t, k_t, v_t)
        self.memory_requirement = (
            self._weight_memory_requirement()
            + self.K_cache.size * self.K_cache.data_type.word_size
            + self.V_cache.size * self.V_cache.data_type.word_size
        )
        return self._merge_heads(hidden, batch_size, 1)
