# Mars Object Detection - YOLOv8

Terrain detection for Mars rover imagery (Soil, Bedrock, Sand, Big Rock).

## Quick Start

```cmd
# 1. Convert AI4Mars dataset to YOLO format
python do_it.py

# 2. Train model (120 epochs, ~2-4 hours on CPU)
python train_do_it_2.py

# 3. View predictions
python live_inference_windows.py
```

## Files

- `do_it.py` - Dataset conversion (AI4Mars → YOLO)
- `train_do_it_2.py` - Training script (rock-optimized)
- `live_inference_windows.py` - Interactive viewer
- `archive/` - Old scripts and documentation

## Model

- Architecture: YOLOv8 Nano (2.58M params)
- Input: 256×256 RGB
- Classes: Soil, Bedrock, Sand, Big Rock
- Training: 120 epochs, rock-focused augmentation

## Deployment

Trained model exports to ONNX → TFLite → VNNX for PolarFire FPGA.
See `archive/docs/` for detailed deployment guides.
