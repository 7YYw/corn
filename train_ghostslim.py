"""
玉米病害检测 - YOLO26 算法改进对比实验
改进一：C3k2-Ghost + SlimNeck 轻量化改进

对比实验：
  1. Baseline: YOLO26n (原始模型)
  2. 改进模型: YOLO26-GhostSlim (轻量化改进)
"""

from ultralytics import YOLO
import torch

# ============ 配置参数 ============
DATA_YAML = r"E:\毕业设计\corn-main1\dataset\data.yaml"
EPOCHS = 150
IMGSZ = 640
BATCH = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
PATIENCE = 20  # early stopping patience

print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def train_baseline():
    """训练 Baseline YOLO26n 模型"""
    print("\n" + "=" * 60)
    print("训练 Baseline: YOLO26n")
    print("=" * 60)

    model = YOLO("yolo26n.yaml")
    model = YOLO("yolo26n.pt")  # 使用预训练权重

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
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
    return model


def train_ghostslim():
    """训练改进模型 YOLO26-GhostSlim"""
    print("\n" + "=" * 60)
    print("训练改进模型: YOLO26-GhostSlim (GhostConv + SlimNeck)")
    print("=" * 60)

    # 使用自定义 YAML 配置文件
    model = YOLO(
        r"E:\毕业设计\corn-main1\ultralytics\cfg\models\26\yolo26-ghostslim.yaml"
    )

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name="yolo26_ghostslim",
        project="runs/corn_disease",
        exist_ok=True,
        pretrained=False,  # 自定义模型，无预训练权重
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        val=True,
        amp=True,
        seed=42,
    )
    return model


def train_ghostslim_from_baseline():
    """从 Baseline 预训练权重迁移训练 YOLO26-GhostSlim"""
    print("\n" + "=" * 60)
    print("训练改进模型 YOLO26-GhostSlim (加载 YOLO26n 预训练权重)")
    print("=" * 60)

    model = YOLO(
        r"E:\毕业设计\corn-main1\ultralytics\cfg\models\26\yolo26-ghostslim.yaml"
    )
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name="yolo26_ghostslim_pretrained",
        project="runs/corn_disease",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,  # 使用预训练权重时降低学习率
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        val=True,
        amp=True,
        seed=42,
    )
    return model


def evaluate_models():
    """评估并对比所有训练好的模型"""
    print("\n" + "=" * 60)
    print("模型评估与对比")
    print("=" * 60)

    models_to_test = {
        "YOLO26n (Baseline)": "runs/corn_disease/yolo26n_baseline/weights/best.pt",
        "YOLO26-GhostSlim": "runs/corn_disease/yolo26_ghostslim/weights/best.pt",
    }

    for name, path in models_to_test.items():
        try:
            model = YOLO(path)
            metrics = model.val(
                data=DATA_YAML,
                device=DEVICE,
                name=f"eval_{name.replace(' ', '_').replace('(', '').replace(')', '')}",
                project="runs/corn_disease/eval",
            )
            print(f"\n{name} 评估结果:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        except Exception as e:
            print(f"  {name}: 评估失败 - {e}")


def compare_parameters():
    """对比模型参数量和计算量"""
    print("\n" + "=" * 60)
    print("模型参数量与计算量对比")
    print("=" * 60)

    models_info = {
        "YOLO26n (Baseline)": "yolo26n.yaml",
        "YOLO26-GhostSlim": r"E:\毕业设计\corn-main1\ultralytics\cfg\models\26\yolo26-ghostslim.yaml",
    }

    for name, cfg in models_info.items():
        try:
            model = YOLO(cfg)
            # 模型信息已包含参数量
            info = model.info()
            print(f"\n{name}:")
            print(f"  Parameters: {info[0] / 1e6:.2f}M")
            print(f"  GFLOPs: {info[1]:.1f}")
        except Exception as e:
            print(f"  {name}: 获取信息失败 - {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "all"

    if mode in ("all", "compare_params"):
        compare_parameters()

    if mode in ("all", "baseline"):
        train_baseline()

    if mode in ("all", "ghostslim"):
        train_ghostslim()

    if mode in ("all", "ghostslim_pretrained"):
        train_ghostslim_from_baseline()

    if mode in ("all", "evaluate"):
        evaluate_models()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
