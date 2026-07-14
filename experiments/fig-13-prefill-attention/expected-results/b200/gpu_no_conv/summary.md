# Attention Summary

Longest context: `32768`

| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| A100 FP16 | fp16 | 4.00 | 116.458 | 66.235 | 60.624 | matmul_pair (66.214 ms) | fused_core (60.603 ms) | 1.093x |
| A100 INT8 | int8 | 4.00 | 86.374 | 32.357 | 30.323 | matmul_pair (32.336 ms) | fused_core (30.302 ms) | 1.067x |
| B200 FP16 | fp16 | 32.00 | 38.318 | 11.556 | 8.607 | softmax (11.535 ms) | fused_core (8.586 ms) | 1.343x |
| B200 INT8 | int8 | 32.00 | 33.900 | 11.181 | 4.314 | softmax (11.160 ms) | fused_core (4.293 ms) | 2.592x |
| H100 FP16 | fp16 | 16.00 | 61.851 | 21.820 | 20.224 | matmul_pair (21.799 ms) | fused_core (20.203 ms) | 1.079x |
| H100 INT8 | int8 | 16.00 | 51.894 | 13.211 | 10.111 | softmax (13.190 ms) | fused_core (10.090 ms) | 1.307x |
