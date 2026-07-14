# Attention Summary

Longest context: `32768`

| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| A100 FP16 | fp16 | 4.00 | 116.458 | 66.235 | 60.624 | matmul_pair (66.214 ms) | fused_core (60.603 ms) | 1.093x |
| A100 INT8 | int8 | 4.00 | 86.374 | 32.357 | 30.764 | matmul_pair (32.336 ms) | fused_core (30.743 ms) | 1.052x |
| B200 FP16 | fp16 | 32.00 | 38.318 | 11.556 | 8.607 | softmax (11.535 ms) | fused_core (8.586 ms) | 1.343x |
| B200 INT8 | int8 | 32.00 | 33.900 | 13.142 | 4.810 | softmax (13.121 ms) | fused_core (4.789 ms) | 2.732x |
| H100 FP16 | fp16 | 16.00 | 61.851 | 21.820 | 20.224 | matmul_pair (21.799 ms) | fused_core (20.203 ms) | 1.079x |
| H100 INT8 | int8 | 16.00 | 51.894 | 15.528 | 10.699 | softmax (15.507 ms) | fused_core (10.678 ms) | 1.451x |
