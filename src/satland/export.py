"""Export the EuroSAT classifier and the LEVIR-CD Siamese U-Net to ONNX and CoreML.

Both exported graphs are self-contained, so a browser or mobile client only has to supply
pixels:

- classifier: input ``bands`` (N, 13, 64, 64) float32 with the raw EuroSAT digital numbers
  (reflectance x 10,000, i.e. the uint16 values of the GeoTIFF) -> output ``probs`` (N, 10).
  Scaling, band selection, the training-set standardisation and the softmax are in the graph.
- change model: inputs ``before`` and ``after`` (N, 3, 256, 256) RGB in [0, 1] -> output
  ``probs`` (N, 1, 256, 256), the change probability. ImageNet normalisation and the sigmoid
  are in the graph.

ONNX files have a dynamic batch axis (faster re-scoring of the test sets); CoreML models use
batch 1.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from .bands import REFLECTANCE_SCALE
from .change import load_cd_checkpoint
from .config import IMAGENET_MEAN, IMAGENET_STD
from .models import load_checkpoint

CLS_INPUTS, CD_INPUTS, OUTPUT_NAME = ("bands",), ("before", "after"), "probs"


class ClassifierExport(torch.nn.Module):
    """Raw 13-band digital numbers -> class probabilities."""

    def __init__(self, model: torch.nn.Module, band_idx: list[int], stats: dict):
        super().__init__()
        self.model = model
        self.register_buffer("band_idx", torch.tensor(band_idx, dtype=torch.long))
        mean = torch.tensor(stats["mean"], dtype=torch.float32)[band_idx]
        std = torch.tensor(stats["std"], dtype=torch.float32)[band_idx]
        # (x / 10000 - mean) / std folded into one multiply-add
        self.register_buffer("scale", (1.0 / (REFLECTANCE_SCALE * std)).view(1, -1, 1, 1))
        self.register_buffer("shift", (-mean / std).view(1, -1, 1, 1))
        self.full = band_idx == list(range(13))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.full:
            x = torch.index_select(x, 1, self.band_idx)
        return torch.softmax(self.model(x * self.scale + self.shift), dim=1)


class ChangeExport(torch.nn.Module):
    """Before / after RGB in [0, 1] -> change probability map."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((a - self.mean) / self.std, (b - self.mean) / self.std))


def load_classifier(checkpoint: str | Path) -> tuple[ClassifierExport, dict]:
    model, meta = load_checkpoint(checkpoint)
    if meta["with_indices"]:
        raise ValueError("Export supports band-only classifiers (no '+idx' channels).")
    return ClassifierExport(model, meta["band_idx"], meta["stats"]).eval(), meta


def load_change(checkpoint: str | Path) -> ChangeExport:
    model, _ = load_cd_checkpoint(checkpoint)
    return ChangeExport(model).eval()


def example_inputs(kind: str, batch: int = 1) -> tuple[torch.Tensor, ...]:
    if kind == "cls":
        return (torch.rand(batch, 13, 64, 64) * 3000,)
    return torch.rand(batch, 3, 256, 256), torch.rand(batch, 3, 256, 256)


def input_names(kind: str) -> tuple[str, ...]:
    return CLS_INPUTS if kind == "cls" else CD_INPUTS


def export_onnx(wrapper: torch.nn.Module, kind: str, out: Path, opset: int = 17) -> Path:
    """Trace to a single-file ONNX model with a dynamic batch axis."""
    import onnx

    names = input_names(kind)
    out.parent.mkdir(parents=True, exist_ok=True)
    axes = {n: {0: "batch"} for n in (*names, OUTPUT_NAME)}
    torch.onnx.export(wrapper, example_inputs(kind), str(out), input_names=list(names),
                      output_names=[OUTPUT_NAME], dynamic_axes=axes, opset_version=opset,
                      dynamo=False)
    model = onnx.load(str(out))  # re-save so weights are embedded, not external data
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    return out


def to_fp16(src: Path, out: Path) -> Path:
    """FP16 weights and activations; inputs and output stay float32 for the callers."""
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(out))
    return out


class _Calibration:
    """onnxruntime CalibrationDataReader over feed dicts."""

    def __init__(self, feeds: Iterable[dict[str, np.ndarray]]):
        self._it = iter(feeds)

    def get_next(self):
        return next(self._it, None)


def to_int8(src: Path, out: Path, calibration: Iterable[dict[str, np.ndarray]]) -> Path:
    """Static INT8 (QDQ, per-channel weights) calibrated on real training inputs.

    Everything after the last convolution stays in float: for the classifier that is pooling,
    the linear head and the softmax (tiny, and it keeps the class scores smooth); for the change
    model it is the sigmoid, so the 0.5 threshold is not applied to an 8-bit probability.
    """
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = out.with_name(out.stem + "_prep.onnx")
    quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)
    nodes = list(onnx.load(str(prep)).graph.node)  # topologically sorted
    last_conv = max(i for i, n in enumerate(nodes) if n.op_type == "Conv")
    keep_float = [n.name for n in nodes[last_conv + 1:]]
    quantize_static(str(prep), str(out), _Calibration(calibration),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=keep_float)
    prep.unlink()
    return out


def export_coreml(wrapper: torch.nn.Module, kind: str, out: Path, fp16: bool = True) -> Path:
    """CoreML ML Program, batch 1 (Neural Engine / GPU / CPU chosen by the OS at run time)."""
    import coremltools as ct

    inputs = example_inputs(kind)
    traced = torch.jit.trace(wrapper, inputs)
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced, inputs=[ct.TensorType(name=n, shape=tuple(t.shape))
                        for n, t in zip(input_names(kind), inputs, strict=True)],
        outputs=[ct.TensorType(name=OUTPUT_NAME)], convert_to="mlprogram",
        compute_precision=precision, minimum_deployment_target=ct.target.macOS13)
    mlmodel.save(str(out))
    return out


class Runner:
    """Common interface: ``run(feed) -> probs`` for a dict of float32 arrays (batch first)."""

    name = "runner"
    max_batch = 0  # 0 = any batch size

    def run(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def run_batches(self, feeds: Iterable[dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
        """Concatenated outputs and the mean model time per sample in ms."""
        outs, elapsed, count = [], 0.0, 0
        for feed in feeds:
            n = len(next(iter(feed.values())))
            step = self.max_batch or n
            for s in range(0, n, step):
                part = {k: v[s:s + step] for k, v in feed.items()}
                t0 = time.perf_counter()
                outs.append(np.asarray(self.run(part), dtype=np.float32))
                elapsed += time.perf_counter() - t0
            count += n
        return np.concatenate(outs), 1000 * elapsed / max(count, 1)


class TorchRunner(Runner):
    def __init__(self, wrapper: torch.nn.Module, device: str = "cpu"):
        self.wrapper = wrapper.to(device).eval()
        self.device = device
        self.name = f"pytorch:{device}"

    @torch.no_grad()
    def run(self, feed):
        out = self.wrapper(*(torch.from_numpy(v).to(self.device) for v in feed.values()))
        return out.float().cpu().numpy()  # .cpu() waits for the GPU


class OnnxRunner(Runner):
    def __init__(self, path: str | Path, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.name = f"onnx:{Path(path).stem}"

    def run(self, feed):
        return self.session.run([OUTPUT_NAME], feed)[0]


class CoreMLRunner(Runner):
    max_batch = 1

    def __init__(self, path: str | Path, compute_units: str = "ALL"):
        import coremltools as ct

        self.model = ct.models.MLModel(str(path),
                                       compute_units=getattr(ct.ComputeUnit, compute_units))
        self.name = f"coreml:{compute_units.lower()}"

    def run(self, feed):
        return self.model.predict(feed)[OUTPUT_NAME]
