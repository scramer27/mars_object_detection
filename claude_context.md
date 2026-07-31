    120/120      1.24G      1.104      2.652      1.634         53        256: 81% ━━━━━━━━━╸── 53
    120/120      1.24G      1.103      2.648      1.632         73        256: 83% ━━━━━━━━━╸── 54
    120/120      1.24G      1.109      2.664      1.635         94        256: 84% ━━━━━━━━━━── 55
    120/120      1.24G       1.11      2.664      1.636         78        256: 86% ━━━━━━━━━━── 56
    120/120      1.24G      1.111      2.666      1.636         65        256: 87% ━━━━━━━━━━╸─ 57
    120/120      1.24G      1.113      2.671      1.636         75        256: 89% ━━━━━━━━━━╸─ 58
    120/120      1.24G      1.114      2.685      1.637         70        256: 90% ━━━━━━━━━━╸─ 59
    120/120      1.24G      1.117       2.69      1.638         76        256: 92% ━━━━━━━━━━━─ 60
    120/120      1.24G      1.119      2.694       1.64         76        256: 93% ━━━━━━━━━━━─ 61
    120/120      1.24G      1.121      2.696      1.641         83        256: 95% ━━━━━━━━━━━─ 62
    120/120      1.24G      1.123      2.709      1.641         71        256: 96% ━━━━━━━━━━━╸ 63
    120/120      1.26G      1.125      2.717      1.641         55        256: 98% ━━━━━━━━━━━╸ 64
    120/120      1.26G      1.125      2.717      1.641         55        256: 100% ━━━━━━━━━━━━ 65/65 3.6it/s 18.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 14% ━╸──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 28% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 42% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 57% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 71% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 85% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.6s
                   all        443       1072      0.626      0.415      0.482       0.32

120 epochs completed in 0.965 hours.
Optimizer stripped from /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/last.pt, 6.2MB
Optimizer stripped from /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt, 6.2MB

Validating /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt...
Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.13.0 MPS (Apple M1 Max)
Model summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 14% ━╸──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 28% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 42% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 57% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 71% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 85% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 1.1it/s 6.2s
                   all        443       1072      0.585      0.455      0.486      0.322
                  Soil        201        274      0.762      0.631      0.685      0.495
               Bedrock        282        514      0.668      0.568      0.627      0.435
                  Sand        137        233      0.518      0.457      0.459      0.275
              Big Rock         36         51      0.391      0.164      0.175     0.0827
Speed: 0.1ms preprocess, 1.0ms inference, 0.0ms loss, 4.2ms postprocess per image
Results saved to /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga

✅ Training finished. Loading best checkpoint: /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt

--- Running Test Set Evaluation ---
Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.13.0 MPS (Apple M1 Max)
Model summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
val: Fast image access ✅ (ping: 0.1±0.1 ms, read: 173.0±127.1 MB/s, size: 9.8 KB)
val: Scanning /Users/scramer/Documents/mars_object_detection/data/yolo_mars/labels/test.cache... 443 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 443/443 464.5Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 0% ─────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 1% ─────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 2% ─────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 3% ─────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 4% ╸────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 5% ╸────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 6% ╸────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 7% ╸────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 9% ━────
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 10% ━───
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 11% ━───
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 12% ━───
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 13% ━╸──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 14% ━╸──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 15% ━╸──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 16% ━━──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 18% ━━──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 19% ━━──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 20% ━━──
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 21% ━━╸─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 22% ━━╸─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 24% ━━╸─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 26% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 27% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 29% ━━━─
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 30% ━━━╸
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 31% ━━━╸
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 32% ━━━╸
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 33% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 34% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 36% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 37% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 38% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 39% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 40% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 41% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 43% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 44% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 45% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 46% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 48% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 49% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 51% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 52% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 53% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 55% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 56% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 57% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 58% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 60% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 61% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 62% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 63% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 65% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 66% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 67% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 68% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 69% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 71% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 72% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 73% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 74% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 76% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 77% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 79% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 80% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 81% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 83% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 84% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 85% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 87% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 88% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 89% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 90% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 92% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 93% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 94% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 96% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 97% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 98% ━━━━
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 443/443 48.3it/s 9.2s
                   all        443       1074       0.57      0.485      0.453      0.317
                  Soil        194        244      0.761      0.709      0.694      0.548
               Bedrock        282        511      0.669      0.571      0.561      0.398
                  Sand        150        256      0.528      0.469      0.444      0.268
              Big Rock         43         63      0.324       0.19      0.114     0.0529
Speed: 1.0ms preprocess, 8.0ms inference, 0.0ms loss, 4.4ms postprocess per image
Results saved to /Users/scramer/Documents/mars_object_detection/runs/detect/val-7

📊 Test Results:
   - mAP@50:    45.34%
   - mAP@50-95: 31.67%

--- Exporting to VectorBlox ONNX (Opset 12) ---
Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.13.0 CPU (Apple M1 Max)

PyTorch: starting from '/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt' with input shape (1, 3, 256, 256) BCHW and output shape(s) (1, 8, 1344) (5.9 MB)

ONNX: starting export with onnx 1.22.0 opset 12...
ONNX: slimming with onnxslim 0.1.94...
ONNX: export success ✅ 0.4s, saved as '/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx' (11.6 MB)

Export complete (0.5s)
Results saved to /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx
Predict:         yolo predict task=detect model=/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx imgsz=256 
Validate:        yolo val task=detect model=/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx imgsz=256 data=/Users/scramer/Documents/mars_object_detection/data/yolo_mars/mars.yaml  
Visualize:       https://netron.app
🎉 Success! FPGA-ready ONNX model compiled to:
/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx
(mars_object_detection) scramer@MT-400226 mars_object_detection % 