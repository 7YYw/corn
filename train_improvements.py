"""
玉米病害检测 - YOLO26 算法改进对比实验
改进二：C2PSA内改进DiffAttention + SPPF替换为LSKA

对比实验：
  1. Baseline: YOLO26n (原始模型)
  2. C2PSADiff: YOLO26-C2PSADiff (在C2PSA内使用差分注意力DiffAttention)
  3. LSKA: YOLO26-LSKA (SPPF替换为LSKA大核分解注意力)
  4. Diff+LSKA: YOLO26-Diff-LSKA (两种改进结合)
"""

from ultralytics import YOLO
import torch
import os

# ============ 配置参数 ============
DATA_YAML = r"E:\毕业设计\corn-main1\dataset\data.yaml"
EPOCHS = 150
IMGSZ = 640
BATCH = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"
PATIENCE = 30  # early stopping patience
PROJECT = "runs/corn_disease_improved"

# 模型配置文件路径
CFG_DIR = r"E:\毕业设计\corn-main1\ultralytics\cfg\models\26"

print(f"Device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def create_model(cfg_path, pretrained_weights=None):
    """创建模型，可选加载预训练权重"""
    model = YOLO(cfg_path)
    if pretrained_weights:
        model = YOLO(pretrained_weights)
    return model


def train_model(name, cfg_path, pretrained=False, lr0=0.001):
    """训练指定配置的模型"""
    print("\n" + "=" * 60)
    print(f"训练模型: {name}")
    print(f"配置: {cfg_path}")
    print(f"预训练: {pretrained}")
    print("=" * 60)

    model = create_model(
        cfg_path,
        pretrained_weights="yolo26n.pt" if pretrained else None,
    )

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name=name,
        project=PROJECT,
        exist_ok=True,
        pretrained=pretrained,
        optimizer="AdamW",
        lr0=lr0,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        val=True,
        amp=True,
        seed=42,
    )
    print(f"{name} 训练完成!")
    return model




def train_c2psadiff():
    """训练 C2PSADiff 改进模型 (DiffAttention)"""
    cfg = os.path.join(CFG_DIR, "yolo26-c2psadiff.yaml")
    return train_model(
        name="yolo26_c2psadiff",
        cfg_path=cfg,
        pretrained=False,
        lr0=0.001,
    )


def train_lska():
    """训练 LSKA 改进模型 (SPPF->LSKA)"""
    cfg = os.path.join(CFG_DIR, "yolo26-lska.yaml")
    return train_model(
        name="yolo26_lska",
        cfg_path=cfg,
        pretrained=False,
        lr0=0.001,
    )


def train_diff_lska():
    """训练 Diff+LSKA 组合改进模型"""
    cfg = os.path.join(CFG_DIR, "yolo26-diff-lska.yaml")
    return train_model(
        name="yolo26_diff_lska",
        cfg_path=cfg,
        pretrained=False,
        lr0=0.001,
    )


def train_c2psadiff_pretrained():
    """C2PSADiff 加载 Baseline 预训练权重迁移训练"""
    print("\n" + "=" * 60)
    print("C2PSADiff 迁移学习 (加载 YOLO26n 预训练权重)")
    print("=" * 60)
    cfg = os.path.join(CFG_DIR, "yolo26-c2psadiff.yaml")
    model = YOLO(cfg)
    model = YOLO("yolo26n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name="yolo26_c2psadiff_pretrained",
        project=PROJECT,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        val=True,
        amp=True,
        seed=42,
    )
    print("C2PSADiff 迁移学习完成!")
    return model


def train_ghostslim_diff_lska():
    """训练 GhostSlim + Diff + LSKA 组合模型"""
    print("\n" + "=" * 60)
    print("GhostSlim + Diff + LSKA 组合改进")
    print("=" * 60)
    cfg = os.path.join(CFG_DIR, "yolo26-ghostslim-diff-lska.yaml")
    model = YOLO(cfg)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        name="yolo26_ghostslim_diff_lska",
        project=PROJECT,
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        val=True,
        amp=True,
        seed=42,
    )
    print("GhostSlim+Diff+LSKA 训练完成!")
    return results


def evaluate_models():
    """评估并对比所有训练好的模型"""
    print("\n" + "=" * 60)
    print("所有模型评估与对比")
    print("=" * 60)

    models_to_test = {
        "YOLO26n (Baseline)": "runs/detect/runs/corn_disease/yolo26n_baseline/weights/best.pt",
        "YOLO26-GhostSlim": "runs/detect/runs/corn_disease/yolo26_ghostslim/weights/best.pt",
        "YOLO26-C2PSADiff": f"{PROJECT}/yolo26_c2psadiff/weights/best.pt",
        "YOLO26-LSKA": f"{PROJECT}/yolo26_lska/weights/best.pt",
        "YOLO26-Diff-LSKA": f"{PROJECT}/yolo26_diff_lska/weights/best.pt",
    }

    results = {}
    for name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"\n{name}: 模型文件不存在 - {path}")
            continue
        try:
            model = YOLO(path)
            metrics = model.val(
                data=DATA_YAML,
                device=DEVICE,
                name=f"eval_{name.replace(' ', '_').replace('-', '_')}",
                project=f"{PROJECT}/eval",
            )
            results[name] = {
                "mAP50": metrics.box.map50,
                "mAP50-95": metrics.box.map,
                "Precision": metrics.box.mp,
                "Recall": metrics.box.mr,
            }
            print(f"\n{name} 评估结果:")
            print(f"  mAP50:    {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall:   {metrics.box.mr:.4f}")
        except Exception as e:
            print(f"\n{name}: 评估失败 - {e}")

    # 打印对比表格
    print("\n\n" + "=" * 60)
    print("模型对比总结")
    print("=" * 60)
    print(f"{'模型':<25} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 65)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['mAP50']:<10.4f} {metrics['mAP50-95']:<10.4f} "
              f"{metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f}")


def compare_parameters():
    """对比模型参数量和计算量"""
    print("\n" + "=" * 60)
    print("模型参数量与计算量对比")
    print("=" * 60)

    models_info = {
        "YOLO26n (Baseline)": "yolo26n.yaml",
        "YOLO26-C2PSADiff": os.path.join(CFG_DIR, "yolo26-c2psadiff.yaml"),
        "YOLO26-LSKA": os.path.join(CFG_DIR, "yolo26-lska.yaml"),
        "YOLO26-Diff-LSKA": os.path.join(CFG_DIR, "yolo26-diff-lska.yaml"),
    }

    for name, cfg in models_info.items():
        try:
            model = YOLO(cfg)
            info = model.info()
            print(f"\n{name}:")
            print(f"  Parameters: {info[0] / 1e6:.2f}M")
            print(f"  GFLOPs: {info[1]:.1f}")
        except Exception as e:
            print(f"\n{name}: 获取信息失败 - {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "all"

    if mode in ("all", "compare_params"):
        compare_parameters()

    if mode in ("all", "c2psadiff"):
        train_c2psadiff()

    if mode in ("all", "lska"):
        train_lska()

    if mode in ("all", "diff_lska"):
        train_diff_lska()

    if mode in ("all", "c2psadiff_pretrained"):
        train_c2psadiff_pretrained()

    if mode in ("all", "ghostslim_combined"):
        train_ghostslim_diff_lska()

    if mode in ("all", "evaluate"):
        evaluate_models()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
