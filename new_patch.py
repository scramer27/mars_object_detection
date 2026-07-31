import types
import torch
from ultralytics import YOLO

def raw_forward(self, x):
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return x

model = YOLO("runs/detect/output_clean/mars_yolo_fpga/weights/best.pt")
model.model.model[-1].forward = types.MethodType(raw_forward, model.model.model[-1])
model.export(format="onnx", opset=12, imgsz=256, simplify=True, dynamic=False)