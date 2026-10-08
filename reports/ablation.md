| Model | Bands | Channels | Labels used | Test acc | Macro F1 | Train min |
|---|---|---|---|---|---|---|
| random forest (band stats) | `rgb` | 12 | 100% | 0.7972 | 0.7888 | 0.19 |
| random forest (band stats) | `rgb_nir` | 16 | 100% | 0.8502 | 0.8448 | 0.24 |
| random forest (band stats) | `all13` | 60 | 100% | 0.9098 | 0.9046 | 0.77 |
| ResNet-18 (ImageNet) | `rgb` | 3 | 100% | 0.9831 | 0.9824 | 2.18 |
| ResNet-18 (ImageNet) | `rgb_nir` | 4 | 100% | 0.9861 | 0.9855 | 2.12 |
| ResNet-18 (ImageNet) | `rgb_nir_swir` | 6 | 100% | 0.9848 | 0.9844 | 2.17 |
| ResNet-18 (ImageNet) | `no_rgb` | 10 | 100% | 0.9830 | 0.9822 | 2.28 |
| ResNet-18 (ImageNet) | `all13` | 13 | 100% | 0.9874 | 0.9869 | 2.35 |
| ResNet-18 (ImageNet) | `all13+idx` | 17 | 100% | 0.9861 | 0.9856 | 2.42 |
| ResNet-18 (ImageNet) | `rgb` | 3 | 10% | 0.9665 | 0.9657 | 1.18 |
| ResNet-18 (ImageNet) | `all13` | 13 | 10% | 0.9689 | 0.9678 | 1.32 |
| ResNet-18 (ImageNet) | `rgb` | 3 | 1% | 0.8963 | 0.8938 | 0.77 |
| ResNet-18 (ImageNet) | `all13` | 13 | 1% | 0.8957 | 0.8934 | 0.89 |
