# Model latency per backend

arm64 macOS-27.0.1-arm64-arm-64bit; batch 1; median and 90th percentile of 100 runs after 10 warm-up runs, network only.


## eurosat_all13 (input 1x13x64x64)

| Backend | Size (MB) | Median (ms) | p90 (ms) |
|---|---|---|---|
| pytorch_fp32 (cpu) | 44.9 | 2.95 | 3.11 |
| pytorch_fp32 (mps) | 44.9 | 2.97 | 3.45 |
| onnx_fp32 (onnxruntime CPU) | 44.8 | 1.77 | 2.01 |
| onnx_fp16 (onnxruntime CPU) | 22.4 | 1.80 | 2.04 |
| onnx_int8 (onnxruntime CPU) | 11.3 | 0.63 | 0.75 |
| coreml_fp32 (all) | 44.9 | 1.24 | 1.76 |
| coreml_fp32 (cpu_and_gpu) | 44.9 | 1.24 | 1.76 |
| coreml_fp32 (cpu_only) | 44.9 | 1.43 | 1.53 |
| coreml_fp16 (all) | 22.5 | 0.59 | 1.89 |
| coreml_fp16 (cpu_and_gpu) | 22.5 | 0.95 | 1.20 |
| coreml_fp16 (cpu_only) | 22.5 | 0.72 | 0.79 |

## levircd_siamese (input 1x3x256x256 + 1x3x256x256)

| Backend | Size (MB) | Median (ms) | p90 (ms) |
|---|---|---|---|
| pytorch_fp32 (cpu) | 57.4 | 38.85 | 39.49 |
| pytorch_fp32 (mps) | 57.4 | 11.58 | 11.98 |
| onnx_fp32 (onnxruntime CPU) | 57.3 | 34.13 | 36.35 |
| onnx_fp16 (onnxruntime CPU) | 28.7 | 35.01 | 37.49 |
| onnx_int8 (onnxruntime CPU) | 14.6 | 15.62 | 20.67 |
| coreml_fp32 (all) | 57.4 | 5.07 | 5.18 |
| coreml_fp32 (cpu_and_gpu) | 57.4 | 5.07 | 5.21 |
| coreml_fp32 (cpu_only) | 57.4 | 12.70 | 13.00 |
| coreml_fp16 (all) | 28.7 | 1.55 | 1.86 |
| coreml_fp16 (cpu_and_gpu) | 28.7 | 3.25 | 3.36 |
| coreml_fp16 (cpu_only) | 28.7 | 6.64 | 6.83 |
