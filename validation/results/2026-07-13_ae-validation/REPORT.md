# SCOPE AE Validation Report

Run ID: `2026-07-13_ae-validation`

Result: **68 passed, 0 failed**

| Status | Experiment | Metric | Actual | Paper | Tolerance |
| --- | --- | --- | ---: | ---: | ---: |
| PASS | Figure 13 | b200 fp16 32K speedup | 1.34262 | 1.34 | 0.01 |
| PASS | Figure 13 | b200 int8 32K speedup | 3.04623 | 3.05 | 0.01 |
| PASS | Figure 21 | b200 2048 conversion-fusion speedup | 1.05512 | 1.06 | 0.01 |
| PASS | Figure 21 | b200 32768 conversion-fusion speedup | 1.11496 | 1.11 | 0.01 |
| PASS | Figure 13/21 archive audit | b200 gpu_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 13/21 archive audit | b200 gpu_no_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 13 | awsv4 fp16 32K speedup | 1.33886 | 1.34 | 0.01 |
| PASS | Figure 13 | awsv4 int8 32K speedup | 2.51391 | 2.51 | 0.01 |
| PASS | Figure 21 | awsv4 4096 conversion-fusion speedup | 1.11998 | 1.12 | 0.01 |
| PASS | Figure 21 | awsv4 8192 conversion-fusion speedup | 1.72768 | 1.73 | 0.01 |
| PASS | Figure 21 | awsv4 16384 conversion-fusion speedup | 1.90796 | 1.91 | 0.01 |
| PASS | Figure 21 | awsv4 32768 conversion-fusion speedup | 1.97158 | 1.97 | 0.01 |
| PASS | Figure 13/21 archive audit | awsv4 aws_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 13/21 archive audit | awsv4 aws_no_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 13 | tpuv6e fp16 32K speedup | 1.6953 | 1.7 | 0.01 |
| PASS | Figure 13 | tpuv6e int8 32K speedup | 2.80562 | 2.81 | 0.01 |
| PASS | Figure 21 | tpuv6e 2048 conversion-fusion speedup | 1.09507 | 1.1 | 0.01 |
| PASS | Figure 21 | tpuv6e 32768 conversion-fusion speedup | 1.46334 | 1.46 | 0.01 |
| PASS | Figure 13/21 archive audit | tpuv6e tpu_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 13/21 archive audit | tpuv6e tpu_no_conv all reproduced latency rows | 0 | 0 | 1e-12 |
| PASS | Figure 14 | FP16 awsv4 32768 E2E speedup | 1.20665 | 1.207 | 0.005 |
| PASS | Figure 14 | FP16 b200 32768 E2E speedup | 1.18341 | 1.183 | 0.005 |
| PASS | Figure 14 | FP16 tpuv6e 32768 E2E speedup | 1.47295 | 1.473 | 0.005 |
| PASS | Figure 14 | FP16 awsv4 524288 E2E speedup | 1.32874 | 1.329 | 0.005 |
| PASS | Figure 14 | FP16 b200 524288 E2E speedup | 1.34056 | 1.341 | 0.005 |
| PASS | Figure 14 | FP16 tpuv6e 524288 E2E speedup | 1.68136 | 1.681 | 0.005 |
| PASS | Figure 14 | INT8 awsv4 524288 E2E speedup | 1.27859 | 1.28 | 0.005 |
| PASS | Figure 14 | INT8 b200 524288 E2E speedup | 2.68922 | 2.69 | 0.005 |
| PASS | Figure 14 | INT8 tpuv6e 524288 E2E speedup | 1.90835 | 1.91 | 0.005 |
| PASS | Figure 15 | B300 FP16 512K attention speedup | 1.08618 | 1.09 | 0.01 |
| PASS | Figure 15 | B300 FP16 512K E2E speedup | 1.08184 | 1.08 | 0.01 |
| PASS | Figure 15 | B300 INT8 512K attention speedup | 1.94252 | 1.94 | 0.01 |
| PASS | Figure 15 | B300 INT8 512K E2E speedup | 1.89919 | 1.9 | 0.01 |
| PASS | Figure 14 archive audit | all 54 paper CSV rows | 0 | 0 | 1e-12 |
| PASS | Figure 15 archive audit | all 18 paper CSV rows | 1.33227e-15 | 0 | 1e-12 |
| PASS | Table 3 | customsa 2048 throughput TFLOP/s | 1130.86 | 1130.86 | 0.01 |
| PASS | Table 3 | customsa 4096 throughput TFLOP/s | 1526.51 | 1526.51 | 0.01 |
| PASS | Table 3 | customsa 8192 throughput TFLOP/s | 1672.82 | 1672.82 | 0.01 |
| PASS | Table 3 | customsa 16384 throughput TFLOP/s | 1713.89 | 1713.89 | 0.01 |
| PASS | Table 3 | illm 2048 throughput TFLOP/s | 641.65 | 641.65 | 0.01 |
| PASS | Table 3 | illm 4096 throughput TFLOP/s | 746.359 | 746.36 | 0.01 |
| PASS | Table 3 | illm 8192 throughput TFLOP/s | 772.036 | 772.04 | 0.01 |
| PASS | Table 3 | illm 16384 throughput TFLOP/s | 782.084 | 782.08 | 0.01 |
| PASS | Table 3 | intattention 2048 throughput TFLOP/s | 888.732 | 888.73 | 0.01 |
| PASS | Table 3 | intattention 4096 throughput TFLOP/s | 1093.3 | 1093.3 | 0.01 |
| PASS | Table 3 | intattention 8192 throughput TFLOP/s | 1170.41 | 1170.41 | 0.01 |
| PASS | Table 3 | intattention 16384 throughput TFLOP/s | 1104.6 | 1104.6 | 0.01 |
| PASS | Figure 18 | SCOPE minimum per-PE area overhead | 1.08557 | 1.09 | 0.01 |
| PASS | Figure 18 | SCOPE maximum per-PE area overhead | 1.43592 | 1.44 | 0.01 |
| PASS | Figure 18 | SCOPE minimum per-PE power overhead | 1.17857 | 1.18 | 0.01 |
| PASS | Figure 18 | SCOPE maximum per-PE power overhead | 1.33696 | 1.34 | 0.01 |
| PASS | Figure 19 | SCNA-8 area reduction geomean | 12.7822 | 12.8 | 0.1 |
| PASS | Figure 19 | SCNA-8 power reduction geomean | 9.46197 | 9.5 | 0.1 |
| PASS | Figures 18/19 | rendered hardware figures | True | present | n/a |
| PASS | RTL | generated Pinnacle N=8 Verilog | 72 Verilog files | present | n/a |
| PASS | Synthesis evidence | paper-matching report sets extracted | 112 rows | present | n/a |
| PASS | Synthesis evidence | archived Design Compiler area reports | 112 files | present | n/a |
| PASS | Synthesis evidence | archived Design Compiler power reports | 112 files | present | n/a |
| PASS | Synthesis evidence | archived Design Compiler timing reports | 112 files | present | n/a |
| PASS | Synthesis evidence | Synopsys DC version recorded in reports | True | present | n/a |
| PASS | Synthesis evidence | SCOPE INT8 synthesized array sizes | 7 sizes | present | n/a |
| PASS | Synthesis evidence | corrected FSA FP8 paper hierarchy source | area=mesh_1_2, power=mesh_3_3 | present | n/a |
| PASS | Synthesis evidence | TSMC 28 nm library recorded in timing report | True | present | n/a |
| PASS | RTL provenance | selected paper RTL directories indexed | 112 directories | present | n/a |
| PASS | RTL provenance | retained upload ZIP byte matches | 108 exact matches | present | n/a |
| PASS | RTL provenance | overwritten ZIP caveats explicitly indexed | 4 caveats | present | n/a |
| PASS | Figures 18/19 | report-to-fit CSV | 0 | 0 | 1e-12 |
| PASS | Figures 18/19 | 32x32 derived CSV | 0 | 0 | 1e-12 |
