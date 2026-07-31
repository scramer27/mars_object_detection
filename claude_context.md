mars_object_detection) scramer@MT-400226 mars_object_detection % git pull
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 3 (delta 1), reused 3 (delta 1), pack-reused 0 (from 0)
Unpacking objects: 100% (3/3), 1.41 KiB | 289.00 KiB/s, done.
From https://github.com/scramer27/mars_object_detection
   f7cb072..344c164  main       -> origin/main
Updating f7cb072..344c164
Fast-forward
 claude_context.md | 74 +++++++++++++++++++++++++++++++++++++++++++++++++++---------
 1 file changed, 63 insertions(+), 11 deletions(-)
(mars_object_detection) scramer@MT-400226 mars_object_detection % python3 -c "
from ultralytics import YOLO
m = YOLO('runs/detect/output_clean/mars_yolo_fpga/weights/best.pt')
print(m.model.yaml.get('nc'), m.names)
"
4 {0: 'Soil', 1: 'Bedrock', 2: 'Sand', 3: 'Big Rock'}
(mars_object_detection) scramer@MT-400226 mars_object_detection % python new_patch.py

Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.13.0 CPU (Apple M1 Max)
YOLO11n summary (fused): 101 layers, 2,582,932 parameters, 0 gradients, 1.0 GFLOPs

PyTorch: starting from 'runs/detect/output_clean/mars_yolo_fpga/weights/best.pt' with input shape (1, 3, 256, 256) BCHW and output shape(s) ((1, 68, 32, 32), (1, 68, 16, 16), (1, 68, 8, 8)) (5.2 MB)

ONNX: starting export with onnx 1.22.0 opset 12...
ONNX: slimming with onnxslim 0.1.94...
ONNX: export success ✅ 0.6s, saved as 'runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx' (9.9 MB)

Export complete (0.7s)
Results saved to /Users/scramer/Documents/mars_object_detection/runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx
Predict:         yolo predict task=detect model=runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx imgsz=256 
Validate:        yolo val task=detect model=runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx imgsz=256 data=data/yolo_mars/mars.yaml  
Visualize:       https://netron.app
(mars_object_detection) scramer@MT-400226 mars_object_detection % python3 -c "
import onnx
m = onnx.load('runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx')
print('Outputs:', [o.name for o in m.graph.output])
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(o.name, dims)
"
Outputs: ['output0', '493', '514']
output0 [1, 68, 32, 32]
493 [1, 68, 16, 16]
514 [1, 68, 8, 8]
(mars_object_detection) scramer@MT-400226 mars_object_detection % 