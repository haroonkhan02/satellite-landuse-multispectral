---
license: mit
library_name: pytorch
tags:
  - remote-sensing
  - sentinel-2
  - multispectral
  - eurosat
  - change-detection
  - levir-cd
  - onnx
---

# Satellite land use (13-band EuroSAT) and building change detection (LEVIR-CD)

Two models from one project. Training code, band ablation and evaluation:
[github.com/haroonkhan02/satellite-landuse-multispectral](https://github.com/haroonkhan02/satellite-landuse-multispectral).
Browser demo: [spaces/haroonx02/satellite-landuse-browser](https://huggingface.co/spaces/haroonx02/satellite-landuse-browser).

- **EuroSAT classifiers**: ImageNet ResNet-18 with a rebuilt first convolution, trained for 8
  epochs on all 13 Sentinel-2 L1C bands (`eurosat_all13`) or on red, green and blue only
  (`eurosat_rgb`). 10 land-use classes, 64×64 patches.
- **LEVIR-CD change model**: Siamese U-Net with a shared ResNet-18 encoder; the decoder gets
  |f_before − f_after| at every scale. Trained 10 epochs on the 256 px tiles of the official
  training split.

## Files

| File | What it is | Size | Test result |
|---|---|---|---|
| `eurosat_all13.pt` | PyTorch checkpoint (`state_dict`, band list, band statistics, config) | 45 MB | accuracy 0.9874 |
| `eurosat_rgb.pt` | Same, RGB only | 45 MB | accuracy 0.9831 |
| `levircd_absdiff.pt` | PyTorch checkpoint (`state_dict`, config) | 57 MB | F1 0.8895 |
| `onnx/eurosat_all13_fp16.onnx` | ONNX, FP16 weights, float32 input/output | 22 MB | accuracy 0.9874 |
| `onnx/eurosat_all13_int8.onnx` | ONNX, static INT8 (QDQ, per-channel, 256 train patches, percentile ranges) | 11 MB | accuracy 0.9876 |
| `onnx/eurosat_rgb_int8.onnx` | Same recipe, RGB model | 11 MB | accuracy 0.9822 |
| `onnx/levircd_siamese_fp16.onnx` | ONNX, FP16 weights | 29 MB | F1 0.8895 |
| `onnx/levircd_siamese_int8.onnx` | ONNX, static INT8 (QDQ, per-channel, 64 train tiles, MinMax) | 15 MB | F1 0.8895 |

**Classifier ONNX:** input `bands`, float32 `(N, 13, 64, 64)`, the raw EuroSAT digital numbers
(TOA reflectance × 10,000, i.e. the GeoTIFF's uint16 values) in EuroSAT band order
B01…B12, B8A (narrow NIR is last). Scaling, band selection, standardisation with the training
statistics and the softmax are inside the graph. Output `probs` `(N, 10)` in the order
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake. The RGB model takes the same 13-band input and selects B04, B03,
B02 itself.

**Change ONNX:** inputs `before` and `after`, float32 `(N, 3, 256, 256)`, RGB in [0, 1].
Output `probs` `(N, 1, 256, 256)`, change probability (threshold 0.5). The model is symmetric:
swapping the dates gives the same map.

In the INT8 files the in-graph input scaling and everything after the last convolution stay
in float. Quantising the raw 13-band input with one 8-bit scale (steps of ~110 DN) erases the
cirrus band and dropped accuracy to 0.866.

## Evaluation

EuroSAT: own stratified 70/10/20 split (committed in the GitHub repo), 5,400 test patches.
The split is random, so patches from the same area can be in train and test, which flatters
every model. LEVIR-CD: official test split as 2,048 tiles of 256 px, change-class F1 pooled
over all pixels. The Hugging Face mirror used stores JPEGs, so results are not directly
comparable with papers. All exported files were re-scored on the full test sets; see
`reports/export.md` in the repo.

## Data and licences

Code and EuroSAT classifier weights: MIT. EuroSAT (Helber et al., IEEE JSTARS 2019; MIT)
contains modified Copernicus Sentinel-2 data. The change model was trained on LEVIR-CD
(Chen & Shi, Remote Sensing 2020), which is released for academic use, so treat
`levircd_absdiff.pt` and the `levircd_siamese_*` files as research-only.
