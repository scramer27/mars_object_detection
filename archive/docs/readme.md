================================================================================
          END-TO-END MARS OBJECT DETECTION PIPELINE DOCUMENTATION
================================================================================

1. SYSTEM ARCHITECTURE & SETUP
--------------------------------------------------------------------------------
- Target Hardware:     VectorBlox FPGA (Fixed-memory mapping, static tensors)
- Workstation Device:  Apple Silicon (Apple M1 Max, MPS Acceleration)
- Model Architecture:  YOLO11 Nano (yolo11n.pt)
- Model Parameters:    2,582,932 (~2.58M)
- Compute Complexity:  6.3 GFLOPs
- Input Dimensions:    256 x 256 x 3 (Optimized for low-latency onboard inference)


2. TRAINING STRATEGY & CONFIGURATION
--------------------------------------------------------------------------------
- Epochs:              120
- Batch Size:          32
- Custom Loss Weights:
  * Box Loss Weight (box = 9.5): Forces high localization precision on small 
    features (rocks/soil boundaries).
  * Class Loss Weight (cls = 1.5): Balances loss signals across classes.
- Total Training Time: 1.237 hours (~74.2 minutes) on Apple M1 Max GPU (MPS)


3. EVALUATION METRICS & RESULTS SUMMARY
--------------------------------------------------------------------------------
Overall Performance Metrics (Held-Out Test Set):
- Precision (P):       0.576 (57.6% of predicted boxes were correct)
- Recall (R):          0.482 (48.2% of ground-truth targets detected)
- mAP@50:              42.72% (Mean Average Precision at IoU = 0.50)
- mAP@50-95:           30.40% (Strict average mAP across IoU 0.50 to 0.95)

Class Performance Breakdown:
+------------+-----------+--------+---------+------------+
| Class      | Precision | Recall | mAP@50  | mAP@50-95  |
+------------+-----------+--------+---------+------------+
| Soil       | 0.810     | 0.701  | 68.0%   | 54.4%      |
| Bedrock    | 0.666     | 0.562  | 52.1%   | 38.2%      |
| Sand       | 0.595     | 0.477  | 42.4%   | 25.1%      |
| Big Rock   | 0.231     | 0.190  |  8.4%   |  3.8%      |
+------------+-----------+--------+---------+------------+
| ALL        | 0.576     | 0.482  | 42.7%   | 30.4%      |
+------------+-----------+--------+---------+------------+

Note: The "Big Rock" class significantly impacted overall metrics due to 
severe class imbalance (only 63 test instances vs 511 for Bedrock).


4. VECTORBLOX FPGA EXPORT CONSTRAINTS
--------------------------------------------------------------------------------
- Export Format:       ONNX
- Opset Version:       12 (Mandatory for VectorBlox compiler support)
- Tensor Allocation:   dynamic=False (Prevents dynamic memory allocation on BRAM)
- Graph Optimization:  simplify=True (Removes redundant tensor ops via onnxslim)


5. WORKFLOW ACCELERATION TECHNIQUES
--------------------------------------------------------------------------------
1. Silicon Acceleration: 
   Switched PyTorch execution to Apple Silicon MPS (Metal Performance Shaders), 
   reducing per-image validation latencies down to ~1.1 ms.

2. Side-by-Side Viewer UI Optimization:
   Deferred image scaling to GPU memory buffers prior to vector text rendering.
   Rendered high-DPI bounding boxes directly onto scaled 700x700 viewports to 
   eliminate font blurriness.

3. Label Disk Caching:
   Dataset annotations scanned and pre-cached (test.cache) at 464.5 M-items/sec, 
   removing I/O bottlenecks during validation runs.
================================================================================