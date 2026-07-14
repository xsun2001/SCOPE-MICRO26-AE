# Attention Summary

Longest context: `32768`

| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| TPUv3 FP16 | fp16 | 32.00 | 418.475 | 289.146 | 216.778 | matmul_pair (288.846 ms) | fused_core (216.478 ms) | 1.334x |
| TPUv3 INT8 | int8 | 32.00 | 306.152 | 143.634 | 108.539 | matmul_pair (143.334 ms) | fused_core (108.239 ms) | 1.323x |
| TPUv4 FP16 | fp16 | 64.00 | 362.271 | 129.593 | 96.685 | matmul_pair (129.293 ms) | fused_core (96.385 ms) | 1.340x |
| TPUv4 INT8 | int8 | 64.00 | 356.981 | 128.610 | 96.427 | matmul_pair (128.310 ms) | fused_core (96.127 ms) | 1.334x |
| TPUv5e FP16 | fp16 | 32.00 | 514.407 | 181.310 | 135.239 | matmul_pair (181.010 ms) | fused_core (134.939 ms) | 1.341x |
| TPUv5e INT8 | int8 | 32.00 | 374.148 | 90.123 | 67.769 | matmul_pair (89.823 ms) | fused_core (67.469 ms) | 1.330x |
| TPUv5p FP16 | fp16 | 32.00 | 175.651 | 77.876 | 58.131 | matmul_pair (77.576 ms) | fused_core (57.831 ms) | 1.340x |
| TPUv5p INT8 | int8 | 32.00 | 115.892 | 38.795 | 29.215 | matmul_pair (38.495 ms) | fused_core (28.915 ms) | 1.328x |
| TPUv6e FP16 | fp16 | 128.00 | 214.050 | 69.646 | 41.082 | matmul_pair (69.346 ms) | fused_core (40.782 ms) | 1.695x |
| TPUv6e INT8 | int8 | 128.00 | 180.116 | 38.877 | 20.691 | softmax (38.577 ms) | fused_core (20.391 ms) | 1.879x |
