"""YOLOv8n / YOLO11n / YOLO26n 对比训练 + 全面评估 (dataset_balanced)"""
from ultralytics import YOLO

DATA = r"E:\毕业设计\corn-main1\dataset_balanced\data.yaml"
EPOCHS = 150
BATCH = 8
IMSZ = 640
PROJECT = "runs/compare8_11_26"

names = ['blight', 'rust', 'gray_spot', 'health']

models = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}

results_summary = {}

def main():
    for name, weights in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        model = YOLO(weights)
        model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMSZ,
            batch=BATCH,
            device=0,
            patience=30,
            name=name,
            project=PROJECT,
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
            verbose=True,
        )

        # 全面评估
        print(f"\n>>> Evaluating {name} ...")
        metrics = model.val(data=DATA, verbose=False)

        p, r, map50, map50_95 = metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map
        ap_class = metrics.box.ap_class_index
        ap_per_class = metrics.box.ap

        results_summary[name] = {
            "precision": p,
            "recall": r,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "per_class_mAP50": {},
        }

        print(f"\n{'='*60}")
        print(f"  {name} 评估结果")
        print(f"{'='*60}")
        print(f"  Precision:     {p:.4f}")
        print(f"  Recall:        {r:.4f}")
        print(f"  mAP50:         {map50:.4f}")
        print(f"  mAP50-95:      {map50_95:.4f}")
        print(f"  F1-score:      {2*p*r/(p+r):.4f}")
        print(f"\n  各类别 AP50:")
        for i, cls_idx in enumerate(ap_class):
            cls_name = names[int(cls_idx)] if int(cls_idx) < len(names) else f"class_{cls_idx}"
            ap = ap_per_class[i][0]  # AP50
            results_summary[name]["per_class_mAP50"][cls_name] = ap
            print(f"    {cls_name}: {ap:.4f}")

    # 最终对比表
    print(f"\n\n{'='*70}")
    print(f"  最终对比汇总 (dataset_balanced)")
    print(f"{'='*70}")
    header = f"{'模型':<12} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>8} {'F1':>8}"
    print(f"  {header}")
    print(f"  {'-'*52}")
    for name in models.keys():
        r = results_summary[name]
        f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"])
        print(f"  {name:<12} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['mAP50']:>8.4f} {r['mAP50-95']:>8.4f} {f1:>8.4f}")

    print(f"\n  各类别 AP50 对比:")
    cls_header = f"{'模型':<12}" + "".join([f"{c:>14}" for c in names])
    print(f"  {cls_header}")
    print(f"  {'-'*12}{'-'*14*len(names)}")
    for name in models.keys():
        vals = results_summary[name]["per_class_mAP50"]
        line = f"  {name:<12}"
        for c in names:
            line += f"{vals.get(c, 0):>14.4f}"
        print(line)
    print(f"\n{'='*70}")

if __name__ == '__main__':
    main()
