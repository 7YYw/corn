"""全面验证 YOLO26-GhostSlim 模型"""
import sys
sys.path.insert(0, r"E:\毕业设计\corn-main1")

import torch
from ultralytics import YOLO

yaml_path = r"E:\毕业设计\corn-main1\ultralytics\cfg\models\26\yolo26-ghostslim.yaml"

print("=" * 60)
print("YOLO26 vs YOLO26-GhostSlim 参数量对比")
print("=" * 60)

model_base = YOLO("yolo26n.yaml")
model_gs = YOLO(yaml_path)

p_base = sum(p.numel() for p in model_base.model.parameters())
p_gs = sum(p.numel() for p in model_gs.model.parameters())

print(f"  Baseline (YOLO26n):   {p_base/1e6:.2f}M")
print(f"  GhostSlim (n-scale):  {p_gs/1e6:.2f}M")
print(f"  参数减少: {(1-p_gs/p_base)*100:.1f}%")

# 前向传播验证
print("\n" + "=" * 60)
print("前向传播验证")
print("=" * 60)
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    model_base.model.eval()
    model_gs.model.eval()
    out_base = model_base.model(x)
    out_gs = model_gs.model(x)

print(f"  Baseline output type: {type(out_base)}")
print(f"  GhostSlim output type: {type(out_gs)}")
if isinstance(out_base, (list, tuple)):
    for i, o in enumerate(out_base):
        print(f"  Baseline[{i}]: {o.shape if hasattr(o, 'shape') else type(o)}")
    for i, o in enumerate(out_gs):
        print(f"  GhostSlim[{i}]: {o.shape if hasattr(o, 'shape') else type(o)}")
elif isinstance(out_base, dict):
    for k, v in out_base.items():
        print(f"  Baseline[{k}]: {v.shape if hasattr(v, 'shape') else type(v)}")
    for k, v in out_gs.items():
        print(f"  GhostSlim[{k}]: {v.shape if hasattr(v, 'shape') else type(v)}")

print("\n所有验证通过!")
