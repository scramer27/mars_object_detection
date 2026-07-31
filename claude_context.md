Confirmed — this worked exactly as intended.

## Sanity-checking the output shapes

`68 = 4×reg_max(16) + nc(4)` — that matches YOLO11's DFL box regression (4 sides × 16 bins) + 4 class logits, per anchor point, at three scales (32×32 = stride 8, 16×16 = stride 16, 8×8 = stride 32 for a 256×256 input). This is exactly the raw pre-decode format we wanted — no flatten, no concat-across-scales, no DFL softmax, no class sigmoid. Should compile cleanly through VectorBlox now.

## Push it to the payload

```bash
cp runs/detect/output_clean/mars_yolo_fpga/weights/best.onnx fpga_payload/best.onnx
git add fpga_payload/best.onnx
git commit -m "Re-export best.onnx with raw Detect head (3 scale outputs, no flatten/decode) for VectorBlox compatibility"
git push
```

## ⚠️ Before you touch WSL — this changes more than the compile step

The original plan treated this as a drop-in replacement, but it isn't quite — the model's **output contract changed entirely**, from one decoded tensor to three raw per-scale tensors. That has knock-on effects on things already built or planned:

1. **`build_fpga_payload.py`** — if this script previously used the old `best.onnx` (single decoded output) to generate expected/ground-truth outputs for the `.bin` test payloads, it needs to be re-run against the new model, and any decode/postprocessing logic in it now needs to do the DFL decode + class sigmoid + NMS itself, since the model no longer does it.
2. **`yoloInfer.py -v 11`** — the `-v 11` flag likely already implements this exact raw-output decode (that's probably *why* the flag exists — Microchip's example script is built to consume YOLO11's raw per-scale format, not a fully decoded one). This is good news — it means you're now aligned with what the tool expects, rather than fighting it. But worth confirming the output tensor **naming/order** it expects matches `['output0', '493', '514']` at `32×32, 16×16, 8×8` — those `493`/`514` names are ONNX's auto-generated node IDs and could shift between export runs. If `yoloInfer.py` selects outputs by name rather than by position/shape, this could silently break. Worth grepping for how it identifies scale outputs:

```bash
grep -n "output0\|reg_max\|def.*decode\|stride" $VBX_SDK/example/python/yoloInfer.py
```

3. **The onnxruntime comparison script** (still not built) — this now needs real decode logic (DFL bin expectation → distance, anchor point + stride → box, sigmoid on class logits, NMS) to turn either model's raw outputs into comparable boxes. This is a meaningfully bigger script than "run both, diff the boxes" — flagging it now so it's built correctly the first time rather than assumed trivial.

## Immediate next steps

1. `git push` the new `best.onnx` (above).
2. Switch to WSL, `git pull`.
3. Re-run:
```bash
onnx2tf -i fpga_payload/best.onnx -o fpga_payload/tflite_out --output_integer_quantized_tflite -cotof
tflite_preprocess fpga_payload/tflite_out/best_full_integer_quant.tflite --scale 255
vnnx_compile -s V1000 -c ncomp -t fpga_payload/tflite_out/best_full_integer_quant.pre.tflite -o fpga_payload/best.vnnx
```
4. Before running `yoloInfer.py`, do the `grep` above to confirm how it maps outputs to scales — paste the result here and I'll help interpret it if it's not obvious.