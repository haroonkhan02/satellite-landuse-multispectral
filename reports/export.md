# Exported models re-scored on the test sets

PyTorch reference on mps (torch 2.14.1); ONNX on onnxruntime 1.31.1 CPU; CoreML with compute units ALL. Latency here is throughput while scoring (ONNX in batches, CoreML batch 1); see benchmark.md for batch-1 latency.


## eurosat_all13 (5400 test patches)

| Format | MB | Accuracy | Macro F1 | Errors | Same label as PyTorch | Max prob diff |
|---|---|---|---|---|---|---|
| pytorch | 44.9 | 0.9874 | 0.9869 | 68 | – | – |
| onnx_fp32 | 44.8 | 0.9874 | 0.9869 | 68 | 100.00 % | 2.1e-06 |
| onnx_fp16 | 22.4 | 0.9874 | 0.9869 | 68 | 100.00 % | 0.0048 |
| onnx_int8 | 11.3 | 0.9876 | 0.9871 | 67 | 99.87 % | 0.25 |
| coreml_fp32 | 44.9 | 0.9874 | 0.9869 | 68 | 100.00 % | 3.6e-05 |
| coreml_fp16 | 22.5 | 0.9874 | 0.9869 | 68 | 100.00 % | 0.0079 |

## eurosat_rgb (5400 test patches)

| Format | MB | Accuracy | Macro F1 | Errors | Same label as PyTorch | Max prob diff |
|---|---|---|---|---|---|---|
| pytorch | 44.8 | 0.9831 | 0.9824 | 91 | – | – |
| onnx_fp32 | 44.7 | 0.9831 | 0.9824 | 91 | 100.00 % | 3e-06 |
| onnx_fp16 | 22.4 | 0.9831 | 0.9824 | 91 | 100.00 % | 0.0061 |
| onnx_int8 | 11.3 | 0.9822 | 0.9815 | 96 | 99.57 % | 0.68 |
| coreml_fp32 | 44.7 | 0.9831 | 0.9824 | 91 | 100.00 % | 5.3e-05 |
| coreml_fp16 | 22.4 | 0.9833 | 0.9826 | 90 | 99.98 % | 0.014 |

## levircd_siamese (2048 test tiles)

| Format | MB | F1 | IoU | Precision | Recall | Pixels flipped vs PyTorch | Max prob diff |
|---|---|---|---|---|---|---|---|
| pytorch | 57.4 | 0.8895 | 0.8010 | 0.8914 | 0.8876 | – | – |
| onnx_fp32 | 57.3 | 0.8895 | 0.8010 | 0.8914 | 0.8876 | 0.000 % | 0.00025 |
| onnx_fp16 | 28.7 | 0.8895 | 0.8010 | 0.8915 | 0.8876 | 0.001 % | 0.026 |
| onnx_int8 | 14.6 | 0.8895 | 0.8010 | 0.8930 | 0.8860 | 0.047 % | 0.58 |
| coreml_fp32 | 57.4 | 0.8895 | 0.8010 | 0.8914 | 0.8876 | 0.000 % | 0.00025 |
| coreml_fp16 | 28.7 | 0.8895 | 0.8010 | 0.8914 | 0.8877 | 0.002 % | 0.058 |
