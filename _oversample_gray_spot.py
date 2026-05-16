# -*- coding: utf-8 -*-
"""
Oversample gray_spot class to balance the dataset.
Uses PIL for I/O (handles Unicode paths on Windows).
"""

import os
import numpy as np
import random
from PIL import Image, ImageEnhance

random.seed(42)
np.random.seed(42)

IMG_DIR = 'E:/毕业设计/corn-main1/dataset/train/images'
LABEL_DIR = 'E:/毕业设计/corn-main1/dataset/train/labels'

TARGET_CLASS = 2  # gray_spot
class_names = ['blight', 'common_rust', 'gray_spot', 'health']

# Find all gray_spot images
gray_spot_images = []
for f in os.listdir(LABEL_DIR):
    if not f.endswith('.txt'):
        continue
    lf_path = os.path.join(LABEL_DIR, f)
    with open(lf_path) as lf:
        lines = [l.strip() for l in lf if l.strip()]
    if not lines:
        continue
    cls = int(lines[0].split()[0])
    if cls == TARGET_CLASS:
        stem = os.path.splitext(f)[0]
        for img_f in os.listdir(IMG_DIR):
            if os.path.splitext(img_f)[0] == stem:
                gray_spot_images.append((stem, img_f, f))
                break

print(f'Found {len(gray_spot_images)} gray_spot images to augment')

generated = 0
for stem, img_f, label_f in gray_spot_images:
    img_path = os.path.join(IMG_DIR, img_f)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f'WARNING: cannot read {img_f}: {e}')
        continue

    w, h = img.size

    # Read label
    label_path = os.path.join(LABEL_DIR, label_f)
    with open(label_path) as f:
        parts = f.read().strip().split()
    cls_str, cx, cy, bw, bh = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    aug = img.copy()
    new_cx, new_cy = cx, cy

    # Random horizontal flip (50%)
    if random.random() < 0.5:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        new_cx = 1.0 - cx

    # Random vertical flip (30%)
    if random.random() < 0.3:
        aug = aug.transpose(Image.FLIP_TOP_BOTTOM)
        new_cy = 1.0 - cy

    # Brightness jitter (80%)
    if random.random() < 0.8:
        factor = 1.0 + random.uniform(-0.15, 0.15)
        aug = ImageEnhance.Brightness(aug).enhance(factor)

    # Contrast jitter (70%)
    if random.random() < 0.7:
        factor = 1.0 + random.uniform(-0.15, 0.15)
        aug = ImageEnhance.Contrast(aug).enhance(factor)

    # Color / Saturation jitter (60%)
    if random.random() < 0.6:
        factor = 1.0 + random.uniform(-0.2, 0.2)
        aug = ImageEnhance.Color(aug).enhance(factor)

    # --- Save ---
    aug_stem = f'{stem}_aug_{generated}'
    aug_img_path = os.path.join(IMG_DIR, f'{aug_stem}.jpg')
    aug.save(aug_img_path, 'JPEG', quality=95)

    aug_label_path = os.path.join(LABEL_DIR, f'{aug_stem}.txt')
    with open(aug_label_path, 'w') as f:
        f.write(f'{cls_str} {new_cx:.6f} {new_cy:.6f} {bw:.6f} {bh:.6f}\n')

    generated += 1
    if generated % 200 == 0:
        print(f'  Progress: {generated}/{len(gray_spot_images)}')

print(f'\nGenerated {generated} augmented gray_spot samples')

# Verify final distribution
print('\n=== FINAL TRAIN CLASS DISTRIBUTION ===')
class_counts = {0:0, 1:0, 2:0, 3:0}
for f in os.listdir(LABEL_DIR):
    if not f.endswith('.txt'):
        continue
    with open(os.path.join(LABEL_DIR, f)) as lf:
        lines = [l.strip() for l in lf if l.strip()]
    if lines:
        cls = int(lines[0].split()[0])
        class_counts[cls] = class_counts.get(cls, 0) + 1

total = sum(class_counts.values())
for cls_id in sorted(class_counts):
    name = class_names[cls_id]
    print(f'  [{cls_id}] {name:20s}: {class_counts[cls_id]:5d} ({class_counts[cls_id]/total*100:.1f}%)')

print(f'\nTotal training samples: {total}')
print(f'Images in directory: {len(os.listdir(IMG_DIR))}')
