# Config Report

This report summarizes every JSON config currently under `LLMCompass/configs/ours/`.

Notes:
- `Chips / device` is `device.compute_chiplet_count`. It is intentionally a per-device number, not top-level `device_count`.
- `Systolic arrays / chip` is `core_count * systolic_array_count`.
- `Vector units / chip` is `core_count * vector_count`.
- LLMCompass uses a single modeled device clock, so `Tensor freq` and `Vector freq` both come from `device.frequency_Hz`.
- `Tensor throughput INT8` is the modeled low-precision tensor throughput from `int8_multiplier`; it is not always an official vendor INT8 figure.
- `Vector throughput EXP` is the modeled MUFU / exponential throughput.
- `GEMM / EXP ratio` is `FP16 tensor throughput / EXP throughput`.
- Capacities are shown in GiB when the config stores bytes, and in both GiB and GB when the config stores decimal GB.

| Metric                 | AWSv2          | AWSv3          | AWSv4          | B200                   | GA100                | H100                 | TPUv3          | TPUv4          | TPUv5e               | TPUv5p         | TPUv6e               |
| ---------------------- | -------------- | -------------- | -------------- | ---------------------- | -------------------- | -------------------- | -------------- | -------------- | -------------------- | -------------- | -------------------- |
| Chips / device         | 1              | 1              | 1              | 1                      | 1                    | 1                    | 1              | 1              | 1                    | 1              | 1                    |
| Systolic arrays / chip | 2              | 8              | 8              | 592                    | 432                  | 528                  | 4              | 8              | 4                    | 8              | 2                    |
| Systolic array size    | 128x128        | 128x128        | 128x128        | 16x16                  | 16x16                | 16x16                | 128x128        | 128x128        | 128x128              | 128x128        | 256x256              |
| Tensor freq            | 2.899 GHz      | 2.544 GHz      | 2.560 GHz      | 1.850 GHz              | 1.410 GHz            | 1.755 GHz            | 0.940 GHz      | 1.050 GHz      | 1.500 GHz            | 1.750 GHz      | 1.750 GHz            |
| Tensor throughput FP16 | 189.989 TFLOPS | 666.894 TFLOPS | 671.089 TFLOPS | 2242.970 TFLOPS        | 311.869 TFLOPS       | 948.879 TFLOPS       | 123.208 TFLOPS | 275.251 TFLOPS | 196.608 TFLOPS       | 458.752 TFLOPS | 917.504 TFLOPS       |
| Tensor throughput INT8 | 379.978 TOPS   | 1333.789 TOPS  | 2684.355 TOPS  | 4485.939 TOPS          | 623.739 TOPS         | 1897.759 TOPS        | 246.415 TOPS   | 275.251 TOPS   | 393.216 TOPS         | 917.504 TOPS   | 1835.008 TOPS        |
| Vector units / chip    | 2              | 8              | 8              | 592                    | 432                  | 528                  | 16             | 16             | 8                    | 16             | 8                    |
| Vector width           | 124 lanes      | 242 lanes      | 240 lanes      | 32 lanes               | 32 lanes             | 32 lanes             | 128 lanes      | 128 lanes      | 128 lanes            | 128 lanes      | 128 lanes            |
| Vector freq            | 2.899 GHz      | 2.544 GHz      | 2.560 GHz      | 1.850 GHz              | 1.410 GHz            | 1.755 GHz            | 0.940 GHz      | 1.050 GHz      | 1.500 GHz            | 1.750 GHz      | 1.750 GHz            |
| Vector throughput ALU  | 4.314 TFLOPS   | 9.850 TFLOPS   | 9.830 TFLOPS   | 70.093 TFLOPS          | 77.967 TFLOPS        | 59.305 TFLOPS        | 3.850 TFLOPS   | 4.301 TFLOPS   | 6.144 TFLOPS         | 14.336 TFLOPS  | 7.168 TFLOPS         |
| Vector throughput EXP  | 4.314 Texp/s   | 9.850 Texp/s   | 39.322 Texp/s  | 4.381 Texp/s           | 2.436 Texp/s         | 3.707 Texp/s         | 1.925 Texp/s   | 2.150 Texp/s   | 3.072 Texp/s         | 7.168 Texp/s   | 3.584 Texp/s         |
| GEMM / EXP ratio       | 44.043:1       | 67.702:1       | 17.067:1       | 512.000:1              | 128.000:1            | 256.000:1            | 64.000:1       | 128.000:1      | 64.000:1             | 64.000:1       | 256.000:1            |
| Memory type            | HBM2           | HBM2           | HBM2           | HBM3e                  | HBM2e                | HBM3                 | HBM2           | HBM2           | HBM                  | HBM            | HBM                  |
| Memory capacity        | 32 GiB         | 96 GiB         | 144 GiB        | 167.64 GiB (180.00 GB) | 74.51 GiB (80.00 GB) | 74.51 GiB (80.00 GB) | 32 GiB         | 32 GiB         | 14.90 GiB (16.00 GB) | 95 GiB         | 29.80 GiB (32.00 GB) |
| Memory bandwidth       | 0.820 TB/s     | 2.900 TB/s     | 4.900 TB/s     | 8.000 TB/s             | 2.039 TB/s           | 3.350 TB/s           | 0.900 TB/s     | 1.200 TB/s     | 0.859 TB/s           | 2.765 TB/s     | 1.600 TB/s           |

## Online Fact Check

Method:
- Checked against current official vendor documentation and product pages as of 2026-04-06.
- The table above remains a report of the current JSON configs. The findings below say whether each row is externally confirmed, inferred, or mismatched versus vendor-published numbers.

What vendor sources generally do confirm:
- Device-level memory capacity and memory bandwidth for all three vendors.
- AWS NeuronCore counts, TensorEngine size, published tensor peak, and some vector / scalar throughput statements.
- Google TPU TensorCore and MXU counts, MXU sizes, published peak BF16 or INT8 throughput, and memory specs.
- NVIDIA published dense tensor throughput either directly or by halving sparse specs where the page states dense is half sparse.

What vendor sources generally do not publish in the same form as this report:
- `Tensor freq` and `Vector freq` for most devices.
- `Vector units / chip`, `Vector width`, `Vector throughput ALU`, `Vector throughput EXP`, and `GEMM / EXP ratio` for NVIDIA and TPU.
- Most NVIDIA `Systolic arrays / chip` counts in this exact form; those are model-layer mappings from Tensor Cores / SM structure, not vendor table fields.

### AWS Neuron

Sources:
- [Trainium / Inferentia2 Architecture Guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html)
- [Trainium2 Architecture Guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html)
- [Trainium3 Architecture Guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html)

Findings:
- `AWSv2`: The 2 TensorEngines per chip, `128x128` TensorEngine size, `32 GiB` HBM, and `820 GB/s` bandwidth are consistent with AWS documentation. The report FP16 tensor peak `189.989 TFLOPS` now matches the published `190 TFLOPS` within about `0.01%`. The report INT8 tensor peak `379.978 TOPS` now matches the published `380 TOPS` within about `0.01%`.
- `AWSv3`: The `8` TensorEngines per chip, `128x128` TensorEngine size, `96 GiB` HBM, and `2.9 TB/s` bandwidth match AWS documentation. The report FP16 tensor peak `666.894 TFLOPS` now matches the published `667 TFLOPS` within about `0.02%`. The report INT8 row is still not directly fact-checkable against an official Trainium2 INT8 peak from these pages; AWS publishes `FP8` and BF16 / FP16 / TF32 figures, so this row should be treated as a project-side low-precision proxy rather than an official INT8 value.
- `AWSv4`: The `8` TensorEngines per chip, `128x128` TensorEngine size, `144 GiB` HBM, and `4.9 TB/s` bandwidth are consistent with AWS documentation. The report FP16 tensor peak `671.089 TFLOPS` now matches the published `671 TFLOPS` within about `0.02%`. The report INT8 row is not an official AWS INT8 number; Trainium3 documentation emphasizes `MXFP8` and `MXFP4`, so this row should be treated strictly as a project-side low-precision proxy.
- `Vector throughput ALU`: AWS does publish vector-engine FP32 throughput at the NeuronCore level. The current report still matches the direction of those statements only partially: `AWSv2` report `4.314 TFLOPS` versus about `5.8 TFLOPS` published FP32 vector total, `AWSv3` report `9.850 TFLOPS` versus about `8.0 TFLOPS`, and `AWSv4` report `9.830 TFLOPS` versus about `9.6 TFLOPS`.
- `Vector throughput EXP`: AWS explicitly documents that Trainium3 adds a dedicated fast exponential path at `4x` the throughput of the prior scalar-engine exponential instruction. The report preserves that architectural story numerically with `AWSv4` exp throughput `39.322 Texp/s`, which is exactly `4x` its modeled ALU throughput, while `AWSv2` and `AWSv3` remain `1:1`. The absolute `Texp/s` numbers themselves are still project-model outputs, not official vendor-published throughputs.

### Google TPU

Sources:
- [TPU v3](https://cloud.google.com/tpu/docs/v3)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)

Findings:
- `TPUv3`: Google documents `2` TensorCores per chip, `2` MXUs per TensorCore, for `4` MXUs per chip total. The report `Systolic arrays / chip = 4` and `128x128` array size match. The report BF16 tensor peak `123.208 TFLOPS` matches the official `123 teraflops` closely. `32 GiB` HBM2 and `900 GB/s` match. The current page does not publish an INT8 peak for TPU v3, so the report `246.415 TOPS` row is a model-side assumption rather than an externally verified official figure.
- `TPUv4`: Google documents `2` TensorCores per chip, `4` MXUs per TensorCore, for `8` MXUs per chip total. The report `Systolic arrays / chip = 8` and `128x128` size match. The report BF16 tensor peak `275.251 TFLOPS` matches the official `275 teraflops` closely. `32 GiB` HBM2 and `1.2 TB/s` match. The official page states `275 teraflops (bf16 or int8)` per chip, and the report INT8 row `275.251 TOPS` now matches that published figure closely.
- `TPUv5e`: Google documents `1` TensorCore per chip with `4` MXUs, matching `Systolic arrays / chip = 4`. The report `128x128` array size, BF16 tensor peak `196.608 TFLOPS`, INT8 tensor peak `393.216 TOPS`, `16 GB` HBM, and `800 GiB/s` memory bandwidth all align closely. The bandwidth row `0.859 TB/s` is the decimal-byte expression of `800 GiB/s`.
- `TPUv5p`: Google documents `2` TensorCores per chip, `4` MXUs per TensorCore, for `8` MXUs per chip total. The report BF16 tensor peak `458.752 TFLOPS` matches the official `459 TFLOPs` closely. The memory row is consistent after unit conversion: the TPU v5p page gives `95 GiB` HBM and `2575 GiBps`, which corresponds to the report `95 GiB` and `2.765 TB/s`. The page does not publish a separate INT8 peak in the same summary table, so the report `917.504 TOPS` row remains a project-side model value.
- `TPUv6e`: Google documents `1` TensorCore per chip with `2` MXUs, so `Systolic arrays / chip = 2` matches. The report `256x256` array size matches. The report BF16 tensor peak `917.504 TFLOPS` and INT8 tensor peak `1835.008 TOPS` both match the official `918 TFLOPs` and `1836 TOPs` closely. Memory capacity also matches after unit conversion: the page gives `32 GB`. The current bandwidth row is low: Google publishes `1638 GiBps`, which is about `1.759 TB/s`, while the report uses `1.600 TB/s`.
- `Vector units / chip`, `Vector width`, `Vector throughput ALU`, `Vector throughput EXP`, and `GEMM / EXP ratio` are not current Google product-sheet fields. Those rows should be treated as LLMCompass modeling abstractions, not official TPU specs.

### NVIDIA

Sources:
- [NVIDIA A100 80GB datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)
- [NVIDIA H100](https://www.nvidia.com/en-eu/data-center/h100/)
- [NVIDIA HGX Platform](https://www.nvidia.com/en-us/data-center/hgx/)
- [NVIDIA HGX AI Factory components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)

Findings:
- `GA100`: The report memory rows match the A100 80GB datasheet once units are handled correctly: the config stores `80.00 GB`, which is `74.51 GiB`, and the official bandwidth is `2039 GB/s`, matching `2.039 TB/s`. The report FP16 tensor peak `311.869 TFLOPS` matches the dense half of NVIDIA's official `624 TFLOPS` sparse BF16 / FP16 spec almost exactly. `Systolic arrays / chip`, `Vector units / chip`, `Vector width`, `Vector throughput ALU`, and `Vector throughput EXP` are not official A100 product-sheet fields; they are model-layer decompositions.
- `H100`: The report memory rows are consistent with official `80 GB HBM3` and `3.35 TB/s`. The report FP16 tensor peak `948.879 TFLOPS` is below the dense half of NVIDIA's official `1979 TFLOPS` sparse BF16 / FP16 figure by about `4.1%` because the config clocking and tensor-core abstraction are slightly conservative. As with GA100, the vector rows and the tensor-array count are model-side mappings rather than directly published H100 specs.
- `B200`: Official NVIDIA material is split across the HGX product page and HGX AI Factory reference docs. Those sources confirm `180 GB HBM3e` per GPU and up to `8 TB/s` bandwidth per GPU. The HGX B200 page gives `36 PFLOPS` sparse BF16 / FP16 Tensor Core throughput for an 8-GPU baseboard and explicitly states dense is half sparse, implying `18 PFLOPS` dense total or `2.25 PFLOPS` per GPU. The report FP16 tensor peak `2242.970 TFLOPS` matches that per-GPU dense value closely, within about `0.3%`. The report INT8 tensor peak `4485.939 TOPS` also matches the dense half of the published `72 POPS` sparse 8-GPU HGX B200 figure closely. The report capacity row uses the exact config value `180.00 GB`, shown as `167.64 GiB (180.00 GB)`.
- For all three NVIDIA configs, the `Tensor freq`, `Vector freq`, `Vector units / chip`, `Vector width`, `Vector throughput ALU`, `Vector throughput EXP`, and `GEMM / EXP ratio` rows are not directly fact-checkable from the cited public product pages. They are LLMCompass-compatible structural abstractions layered on top of official Tensor Core throughput and memory numbers.

### Bottom Line

- Strong matches: `AWSv2` FP16 / INT8 and memory, `AWSv3` FP16 and memory, `AWSv4` FP16 and memory, `GA100` FP16 and memory, `B200` FP16 / INT8 and memory, `TPUv3` BF16 and memory, `TPUv4` BF16 / INT8 and memory, `TPUv5e` BF16 / INT8 and memory, `TPUv5p` BF16 and memory, `TPUv6e` BF16 / INT8.
- Clear mismatches to current vendor docs: `TPUv6e` memory bandwidth.
- Rows that should be read as model-only unless you explicitly re-derive them from vendor microarchitecture documents: `Tensor freq`, `Vector units / chip`, `Vector width`, `Vector throughput ALU`, `Vector throughput EXP`, and `GEMM / EXP ratio`.
