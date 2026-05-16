"""训练 Baseline YOLO26n"""
from ultralytics import YOLO

model = YOLO("yolo26n.yaml")
model = YOLO("yolo26n.pt")

results = model.train(
    data=r"E:\毕业设计\corn-main1\dataset\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
    name="yolo26n_baseline",
    project="runs/corn_disease",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    val=True,
    amp=True,
    seed=42,
)
