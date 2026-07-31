# run_onnx2tf_patched.py
import sys, glob
import numpy as np
import onnx2tf.onnx2tf as o2t

def local_calibration_data():
    """NHWC float32 in [0,255] range, uint8-like scale — matches onnx2tf's
    representative_dataset_gen input expectations, NOT the NCHW layout used
    for the raw ONNX/VNNX input tensor."""
    candidates = glob.glob("data/yolo_mars/images/**/*.jpg", recursive=True)
    if candidates:
        from PIL import Image
        img = Image.open(candidates[0]).convert("RGB").resize((256, 256))
        arr = np.asarray(img, dtype=np.float32)          # (256, 256, 3) — NHWC, no transpose
        arr = arr[None, ...]                              # -> (1, 256, 256, 3)
        return arr
    return np.random.default_rng(0).random((1, 256, 256, 3), dtype=np.float32) * 255.0

o2t.download_test_image_data = local_calibration_data

sys.argv = [
    "onnx2tf",
    "-i", "fpga_payload/best.onnx",
    "-o", "fpga_payload/tflite_out",
    "-oiqt",
    "-cotof",
]
o2t.main()