import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

onnx2tf_main = importlib.import_module("onnx2tf.onnx2tf")


def local_calibration_data():
    """
    Return NHWC float32 calibration data.

    onnx2tf expects NHWC here, even though the source ONNX input is NCHW.
    Prefer real Mars test images; fall back to deterministic synthetic data.
    """
    image_dir = Path("data/yolo_mars/images/test")
    paths = []

    if image_dir.exists():
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(image_dir.glob(extension))

    paths = sorted(paths)[:50]

    if paths:
        samples = []
        for path in paths:
            image = Image.open(path).convert("RGB").resize((256, 256))
            samples.append(np.asarray(image, dtype=np.float32))
        data = np.stack(samples, axis=0)
        print(f"Using {len(samples)} local Mars images for calibration: {data.shape}")
        return data

    print("WARNING: No local images found; using deterministic synthetic calibration data.")
    rng = np.random.default_rng(42)
    return rng.integers(
        0, 256, size=(20, 256, 256, 3), dtype=np.uint8
    ).astype(np.float32)


# Patch the binding actually called inside onnx2tf.py.
onnx2tf_main.download_test_image_data = local_calibration_data

sys.argv = [
    "onnx2tf",
    "-i", "fpga_payload/best.onnx",
    "-o", "fpga_payload/tflite_out",
    "--output_integer_quantized_tflite",
    "-cotof",
]

onnx2tf_main.main()
