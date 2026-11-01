---
title: Satellite Land Use in the Browser
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: 13-band EuroSAT and LEVIR-CD change models, in-browser
models:
  - haroonx02/satellite-landuse-multispectral
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

Two models from one project, exported to ONNX, quantised to INT8 and run client-side with
onnxruntime-web (WebAssembly): a ResNet-18 land-use classifier on all 13 Sentinel-2 bands
(EuroSAT), shown next to an RGB-only version, and a Siamese U-Net for building change detection
(LEVIR-CD). The composites, NDVI and overlays are ported to JavaScript and checked against the
Python pipeline.

Code, training, band ablation and evaluation:
[github.com/haroonkhan02/satellite-landuse-multispectral](https://github.com/haroonkhan02/satellite-landuse-multispectral).
