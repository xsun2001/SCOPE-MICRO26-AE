# Attention Summary

Longest context: `32768`

| case | dtype | ratio | baseline ms | flash ms | customsa ms | flash bottleneck | customsa bottleneck | flash/customsa |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| B200 FP16 | fp16 | 32.00 | 38.318 | 11.556 | 8.607 | softmax (11.535 ms) | fused_core (8.586 ms) | 1.343x |
| B200 INT8 | int8 | 32.00 | 33.900 | 13.142 | 4.810 | softmax (13.121 ms) | fused_core (4.789 ms) | 2.732x |
