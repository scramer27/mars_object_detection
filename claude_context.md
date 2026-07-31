
(mars_object_detection) scramer@MT-400226 mars_object_detection % find . -name "*full_integer_quat
.tflite
dquote> 
(mars_object_detection) scramer@MT-400226 mars_object_detection % find . -name "*full_integer_quat.tflite"
(mars_object_detection) scramer@MT-400226 mars_object_detection % find . -name *.tflite          
zsh: no matches found: *.tflite
(mars_object_detection) scramer@MT-400226 mars_object_detection % find . -name *.tflite
zsh: no matches found: *.tflite
(mars_object_detection) scramer@MT-400226 mars_object_detection % find . -name "mars_yolov8_fpga_full_integer_quant.pre..tflite"
(mars_object_detection) scramer@MT-400226 mars_object_detection % yolo export \    model=runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt \
    format=tflite \
    int8=True \
    data=data/yolo_mars/mars.yaml  
WARNING ⚠️ 'int8' is deprecated and will be removed in the future. Use 'quantize' instead.
WARNING ⚠️ format='tflite' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert/
Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.12.1 CPU (Apple M1 Max)
WARNING ⚠️ LiteRT INT8 export does not support end2end models, disabling end2end branch.
Model summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 1.3 GFLOPs

PyTorch: starting from 'runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt' with input shape (1, 3, 256, 256) BCHW and output shape(s) (1, 8, 1344) (5.9 MB)
LiteRT: collecting INT8 calibration images from 'data=data/yolo_mars/mars.yaml'
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 170.2±69.6 MB/s, size: 7.5 KB)
val: Scanning /Users/scramer/Documents/mars_object_detection/data/yolo_mars/labels/val.cache... 443 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 443/443 142.9Mit/s 0.0s
W0731 14:16:41.420000 56999 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
W0731 14:16:41.437000 56999 torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.

LiteRT: starting export with litert_torch 0.9.2...
(00:00) [START] LiteRT-Torch Convert
WARNING:root:Your model is converted in training mode. Please set the module in evaluation mode with `module.eval()` for better on-device performance and compatibility.
(00:00) [START] LiteRT-Torch Convert > Torch Export: serving_default
(00:00) [START] LiteRT-Torch Convert > Torch Export: serving_default > ExportedProgram Run 
Decompositions
/Users/scramer/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/lib/python3.11/copyreg.py:105: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
(00:00) [ DONE] LiteRT-Torch Convert > Torch Export: serving_default > ExportedProgram Run 
Decompositions (+00:00)
(00:00) [ DONE] LiteRT-Torch Convert > Torch Export: serving_default (+00:00)
(00:00) [START] LiteRT-Torch Convert > Run FX Passes
(00:01) [START] LiteRT-Torch Convert > Run FX Passes > ExportedProgram Run Decompositions
/Users/scramer/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/lib/python3.11/copyreg.py:105: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
(00:01) [ DONE] LiteRT-Torch Convert > Run FX Passes > ExportedProgram Run Decompositions (+00:00)
(00:01) [ DONE] LiteRT-Torch Convert > Run FX Passes (+00:00)
(00:01) [START] LiteRT-Torch Convert > Lower to MLIR: serving_default
(00:01) [START] LiteRT-Torch Convert > Lower to MLIR: serving_default > ExportedProgram Run 
Decompositions
/Users/scramer/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/lib/python3.11/copyreg.py:105: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
  return cls.__new__(cls, *args)
(00:02) [ DONE] LiteRT-Torch Convert > Lower to MLIR: serving_default > ExportedProgram Run 
Decompositions (+00:00)
(00:02) [START] LiteRT-Torch Convert > Lower to MLIR: serving_default > ExportedProgram Run 
Decompositions
(00:02) [ DONE] LiteRT-Torch Convert > Lower to MLIR: serving_default > ExportedProgram Run 
Decompositions (+00:00)
(00:02) [START] LiteRT-Torch Convert > Lower to MLIR: serving_default > Create MLIR Module
(00:04) [ DONE] LiteRT-Torch Convert > Lower to MLIR: serving_default > Create MLIR Module 
(+00:01)
(00:04) [ DONE] LiteRT-Torch Convert > Lower to MLIR: serving_default (+00:02)
(00:04) [START] LiteRT-Torch Convert > Merge MLIR Modules
(00:04) [ DONE] LiteRT-Torch Convert > Merge MLIR Modules (+00:00)
(00:04) [START] LiteRT-Torch Convert > Run LiteRT Converter Passes
(00:04) [ DONE] LiteRT-Torch Convert > Run LiteRT Converter Passes (+00:00)
(00:04) [ DONE] LiteRT-Torch Convert (+00:04)
(00:00) [START] Write Model to 
runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite
(00:00) [ DONE] Write Model to 
runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite (+00:00)
LiteRT: applying static quantization (int8 weights + int8 activations)...
/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ai_edge_litert/interpreter.py:480: UserWarning: Warning: Enabling `experimental_preserve_all_tensors` with the BUILTIN or AUTO op resolver is intended for debugging purposes only. Be aware that this can significantly increase memory usage by storing all intermediate tensors. If you encounter memory problems or are not actively debugging, consider disabling this option.
  warnings.warn(
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Generating Quantization Parameters:: 100%|██████████████████| 256/256 [00:00<00:00, 10835.48it/s]
Applying Transformations to tensors:: 100%|█████████████████| 413/413 [00:00<00:00, 76286.94it/s]
Model name: runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite
Original model size: 11.61 MiB
Quantized model size: 3.15 MiB
Quantization Ratio: 0.27 (3.7x smaller)
Total time: 49.43 ms
WARNING:root:The model runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite already exists in the folder. Overwriting the model since overwrite=True.
LiteRT: export success ✅ 44.1s, saved as 'runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite' (3.2 MB)

Export complete (44.2s)
Results saved to /Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite
Predict:         yolo predict task=detect model=runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite imgsz=256 
Validate:        yolo val task=detect model=runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite imgsz=256 data=/Users/scramer/Documents/mars_object_detection/data/yolo_mars/mars.yaml  
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export
VS Code: view Ultralytics VS Code Extension ⚡ at https://docs.ultralytics.com/integrations/vscode
(mars_object_detection) scramer@MT-400226 mars_object_detection % 