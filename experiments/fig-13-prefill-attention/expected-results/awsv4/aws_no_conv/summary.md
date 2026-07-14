# Attention Summary

Longest context: `32768`

| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| AWSv2 FP16 | fp16 | 44.04 | 202.781 | 186.193 | 140.686 | matmul_pair (185.893 ms) | fused_core (140.386 ms) | 1.323x |
| AWSv2 INT8 | int8 | 44.04 | 154.131 | 92.893 | 70.493 | matmul_pair (92.593 ms) | fused_core (70.193 ms) | 1.318x |
| AWSv3 FP16 | fp16 | 67.70 | 174.903 | 53.664 | 40.081 | matmul_pair (53.364 ms) | fused_core (39.781 ms) | 1.339x |
| AWSv3 INT8 | int8 | 67.70 | 119.575 | 26.781 | 20.191 | matmul_pair (26.481 ms) | fused_core (19.891 ms) | 1.326x |
| AWSv4 FP16 | fp16 | 68.27 | 166.242 | 53.330 | 39.833 | matmul_pair (53.030 ms) | fused_core (39.533 ms) | 1.339x |
| AWSv4 INT8 | int8 | 68.27 | 85.716 | 22.313 | 14.404 | softmax (22.013 ms) | onchip_io (14.104 ms) | 1.549x |
