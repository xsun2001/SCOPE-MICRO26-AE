from software_model.operators import Operator
from software_model.matmul import Matmul
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.attention import (
    MultiHeadAttentionInitComputationTP,
    MultiHeadAttentionAutoRegressionTP,
)
from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from hardware_model.system import System


class _TransformerBlockTPBase(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count

        d = d_model
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)

        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def _bind_attention_aliases(self):
        self.Wq = self.attention.Wq
        self.Wk = self.attention.Wk
        self.Wv = self.attention.Wv
        self.W0 = self.attention.W0
        self.Q_proj = self.attention.Q_proj
        self.K_proj = self.attention.K_proj
        self.V_proj = self.attention.V_proj
        self.Q_reshape = self.attention.Q_reshape
        self.K_reshape = self.attention.K_reshape
        self.V_reshape = self.attention.V_reshape
        self.Q_transpose = self.attention.Q_transpose
        self.K_transpose = self.attention.K_transpose
        self.V_transpose = self.attention.V_transpose
        self.Q_mul_K = self.attention.Q_mul_K
        self.A_softmax = self.attention.A_softmax
        self.A_mul_V = self.attention.A_mul_V
        self.H_transpose = self.attention.H_transpose
        self.H_reshape = self.attention.H_reshape
        self.H_matmul0 = self.attention.H_matmul0
        if hasattr(self.attention, "K_concat"):
            self.K_concat = self.attention.K_concat
        if hasattr(self.attention, "V_concat"):
            self.V_concat = self.attention.V_concat

    def _apply_post_attention(self, hidden: Tensor) -> Tensor:
        hidden = self.layer_norm0(hidden)
        assert hidden.shape[-1] == self.d_model
        if self.device_count > 1:
            hidden = self.allreduce_mha(hidden)
        return hidden

    def _apply_ffn(self, hidden: Tensor) -> Tensor:
        output = self.H_matmul1(hidden, self.W1)
        assert output.shape[-1] == 4 * self.d_model // self.device_count
        output = self.H_gelu(output)
        output = self.H_matmul2(output, self.W2)
        assert output.shape[-1] == self.d_model
        output = self.layer_norm1(output)
        if self.device_count > 1:
            output = self.allreduce_ffn(output)
        return output

    def _matmul_latency(
        self,
        op: Matmul,
        device,
        mode: str,
        compile_mode: str = None,
    ) -> float:
        if mode == "roofline":
            return op.roofline_model(device) + device.compute_module.overhead.matmul
        if mode == "compile":
            return (
                op.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.matmul
            )
        if mode == "gpu":
            return op.run_on_gpu()
        raise ValueError(f"Unsupported transformer measurement mode '{mode}'.")

    def _layernorm_latency(self, device, mode: str, compile_mode: str = None) -> float:
        if mode == "roofline":
            return (
                self.layer_norm0.roofline_model(device)
                + device.compute_module.overhead.layernorm
            )
        if mode == "compile":
            return (
                self.layer_norm0.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.layernorm
            )
        if mode == "gpu":
            return (
                self.layer_norm0.run_on_gpu()
                - self.layer_norm0.gpu_kernel_launch_overhead()
            )
        raise ValueError(f"Unsupported transformer measurement mode '{mode}'.")

    def _gelu_latency(self, device, mode: str, compile_mode: str = None) -> float:
        if mode == "roofline":
            return self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        if mode == "compile":
            return (
                self.H_gelu.compile_and_simulate(device, compile_mode)
                + device.compute_module.overhead.gelu
            )
        if mode == "gpu":
            return self.H_gelu.run_on_gpu()
        raise ValueError(f"Unsupported transformer measurement mode '{mode}'.")

    def _format_breakdown_log(
        self,
        h1_matmul1_latency: float,
        h2_matmul2_latency: float,
        layernorm_latency: float,
        gelu_latency: float,
        allreduce_latency: float,
    ) -> str:
        return (
            f"{self.attention.qkv_latency}, {self.attention.q_mul_k_latency}, "
            f"{self.attention.a_mul_v_latency}, {self.attention.h_matmul0_latency}, "
            f"{h1_matmul1_latency}, {h2_matmul2_latency}, "
            f"{self.attention.softmax_latency}, {layernorm_latency}, {layernorm_latency}, "
            f"{gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        )

    def _print_breakdown(
        self,
        title: str,
        h1_matmul1_latency: float,
        h2_matmul2_latency: float,
        layernorm_latency: float,
        gelu_latency: float,
        allreduce_latency: float,
        attention_total_latency: float,
        allreduce_total_latency: float,
    ):
        print(title)
        print(
            f"{self.attention.qkv_latency}\n{self.attention.q_mul_k_latency}\n"
            f"{self.attention.a_mul_v_latency}\n{self.attention.h_matmul0_latency}\n"
            f"{h1_matmul1_latency}\n{h2_matmul2_latency}\n"
            f"{self.attention.softmax_latency}\n{layernorm_latency}\n"
            f"{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n"
            f"{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{attention_total_latency}\n"
            f"{h1_matmul1_latency + h2_matmul2_latency}\n"
            f"{layernorm_latency * 2}\n"
            f"{gelu_latency}\n"
            f"{allreduce_total_latency}\n"
        )

    def _measure_system_latency(
        self,
        system: System,
        mode: str,
        compile_mode: str = None,
        print_breakdown: bool = False,
    ) -> float:
        device = system.device
        interconnect = system.interconnect

        if mode == "roofline":
            attention_total_latency = self.attention.roofline_model(device)
        elif mode == "compile":
            attention_total_latency = self.attention.compile_and_simulate(
                device,
                compile_mode,
            )
        else:
            raise ValueError(f"Unsupported transformer measurement mode '{mode}'.")

        h1_matmul1_latency = self._matmul_latency(
            self.H_matmul1,
            device,
            mode,
            compile_mode,
        )
        h2_matmul2_latency = self._matmul_latency(
            self.H_matmul2,
            device,
            mode,
            compile_mode,
        )
        layernorm_latency = self._layernorm_latency(device, mode, compile_mode)
        gelu_latency = self._gelu_latency(device, mode, compile_mode)

        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        total_latency = (
            attention_total_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + layernorm_latency * 2
            + gelu_latency
            + allreduce_total_latency
        )

        log = self._format_breakdown_log(
            h1_matmul1_latency,
            h2_matmul2_latency,
            layernorm_latency,
            gelu_latency,
            allreduce_latency,
        )
        if mode == "roofline":
            self.roofline_latency = total_latency
            self.roofline_log = log
            if print_breakdown:
                self._print_breakdown(
                    "Roofline breakdown:",
                    h1_matmul1_latency,
                    h2_matmul2_latency,
                    layernorm_latency,
                    gelu_latency,
                    allreduce_latency,
                    attention_total_latency,
                    allreduce_total_latency,
                )
        else:
            self.latency = total_latency
            self.simluate_log = log

        return total_latency

    def roofline_model(self, system: System):
        return self._measure_system_latency(
            system,
            mode="roofline",
            print_breakdown=True,
        )

    def compile_and_simulate(self, system: System, compile_mode: str):
        return self._measure_system_latency(
            system,
            mode="compile",
            compile_mode=compile_mode,
        )

    def run_on_gpu(self):
        attention_total_latency = self.attention.run_on_gpu()
        h1_matmul1_latency = self._matmul_latency(self.H_matmul1, None, "gpu")
        h2_matmul2_latency = self._matmul_latency(self.H_matmul2, None, "gpu")
        layernorm_latency = self._layernorm_latency(None, "gpu")
        gelu_latency = self._gelu_latency(None, "gpu")
        allreduce_latency = 0
        allreduce_total_latency = 0

        self.latency_on_gpu = (
            attention_total_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
            + layernorm_latency * 2
            + gelu_latency
            + allreduce_total_latency
        )

        self._print_breakdown(
            "breakdown:",
            h1_matmul1_latency,
            h2_matmul2_latency,
            layernorm_latency,
            gelu_latency,
            allreduce_latency,
            attention_total_latency,
            allreduce_total_latency,
        )
        return self.latency_on_gpu


class TransformerBlockInitComputationTP(_TransformerBlockTPBase):
    def __init__(
        self,
        d_model,
        n_heads,
        device_count,
        data_type: DataType,
        attention_variant: str = "trivial",
    ):
        super().__init__(d_model, n_heads, device_count, data_type)
        self.attention = MultiHeadAttentionInitComputationTP(
            d_model,
            n_heads,
            device_count,
            data_type,
            attention_variant=attention_variant,
        )
        self.attention_variant = self.attention.attention_variant
        self._bind_attention_aliases()

    def __call__(self, x: Tensor) -> Tensor:
        hidden = self.attention(x)
        hidden = self._apply_post_attention(hidden)
        output = self._apply_ffn(hidden)
        assert output.shape == [x.shape[0], x.shape[1], self.d_model]
        return output


class TransformerBlockAutoRegressionTP(_TransformerBlockTPBase):
    def __init__(
        self,
        d_model,
        n_heads,
        device_count,
        data_type: DataType,
        attention_variant: str = "trivial",
    ):
        super().__init__(d_model, n_heads, device_count, data_type)
        self.attention = MultiHeadAttentionAutoRegressionTP(
            d_model,
            n_heads,
            device_count,
            data_type,
            attention_variant=attention_variant,
        )
        self.attention_variant = self.attention.attention_variant
        self._bind_attention_aliases()

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        hidden = self.attention(x, seq_len)
        hidden = self._apply_post_attention(hidden)
        output = self._apply_ffn(hidden)
        assert output.shape == [x.shape[0], 1, self.d_model]
        self.memory_requirement = (
            self.attention.memory_requirement
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
        )
        return output


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass
