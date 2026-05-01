#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO train2017 人体关键点 JSON 标签 → YOLO 格式标签转换工具

将 COCO 官方标注文件（person_keypoints_train2017.json 等）转换为 YOLOv5 人体关键点
检测模型所需的 .txt 标签格式。

COCO 关键点格式（person_keypoints_*.json）：
  每个标注包含：
    - bbox: [x_min, y_min, width, height]（像素坐标，左上角原点）
    - keypoints: [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
      其中 vi 为可见性标志：0=未标注，1=已标注但不可见，2=已标注且可见

YOLO 关键点格式（每行一个实例）：
  class_id cx cy bw bh kx1 ky1 kv1 kx2 ky2 kv2 ... kx17 ky17 kv17
  其中：
    - class_id: 类别索引（人=0）
    - cx, cy, bw, bh: 边界框中心坐标和宽高（归一化到 [0, 1]）
    - kxi, kyi: 关键点坐标（归一化到 [0, 1]）
    - kvi: 可见性（0/1/2，不归一化）

COCO 17 个人体关键点顺序：
  0=鼻子, 1=左眼, 2=右眼, 3=左耳, 4=右耳,
  5=左肩, 6=右肩, 7=左肘, 8=右肘,
  9=左腕, 10=右腕, 11=左髋, 12=右髋,
  13=左膝, 14=右膝, 15=左踝, 16=右踝

使用示例：
  # 转换训练集
  python utils/coco2yolo_keypoints.py \\
      --json /path/to/person_keypoints_train2017.json \\
      --output /path/to/labels/train2017

  # 转换验证集
  python utils/coco2yolo_keypoints.py \\
      --json /path/to/person_keypoints_val2017.json \\
      --output /path/to/labels/val2017

  # 跳过没有关键点的实例
  python utils/coco2yolo_keypoints.py \\
      --json /path/to/person_keypoints_train2017.json \\
      --output /path/to/labels/train2017 \\
      --require-keypoints
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# COCO 人体关键点数量（固定为 17）
COCO_NUM_KEYPOINTS = 17


def convert_coco_to_yolo_keypoints(
    json_path: str,
    output_dir: str,
    require_keypoints: bool = False,
    min_area: float = 0.0,
    person_category_id: int = None,
) -> None:
    """将 COCO JSON 格式的人体关键点标注转换为 YOLO 格式的 .txt 标签文件。

    参数：
        json_path (str): COCO 标注文件路径，例如 person_keypoints_train2017.json
        output_dir (str): 输出 .txt 标签文件的目录
        require_keypoints (bool): 若为 True，则跳过 num_keypoints=0 的实例
        min_area (float): 最小边界框面积阈值（归一化面积，跳过小于该值的实例）
        person_category_id (int): 人类别的 ID，若为 None 则自动从 JSON 中检测
    """
    # 加载 COCO JSON 文件
    print(f"正在加载 COCO 标注文件：{json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ——————————————————————————————————————————————
    # 1. 确定人类别 ID
    # ——————————————————————————————————————————————
    if person_category_id is None:
        # 从 categories 中自动查找 "person"
        for cat in coco_data.get("categories", []):
            if cat["name"] == "person":
                person_category_id = cat["id"]
                break
        if person_category_id is None:
            raise ValueError("在 JSON 文件中未找到 'person' 类别，请手动指定 --person-category-id")
    print(f"人体类别 ID：{person_category_id}")

    # ——————————————————————————————————————————————
    # 2. 构建图像 ID → 图像信息的映射
    # ——————————————————————————————————————————————
    id_to_image = {}
    for img_info in coco_data["images"]:
        id_to_image[img_info["id"]] = img_info

    # ——————————————————————————————————————————————
    # 3. 按图像 ID 分组标注
    # ——————————————————————————————————————————————
    img_annotations = defaultdict(list)
    total_skipped = 0
    total_annotations = 0

    for ann in coco_data["annotations"]:
        # 仅处理人类别的标注
        if ann["category_id"] != person_category_id:
            continue

        # 跳过无效边界框（宽或高 <= 0）
        x_min, y_min, bbox_w, bbox_h = ann["bbox"]
        if bbox_w <= 0 or bbox_h <= 0:
            total_skipped += 1
            continue

        # 若要求关键点，跳过没有关键点的实例
        if require_keypoints and ann.get("num_keypoints", 0) == 0:
            total_skipped += 1
            continue

        img_annotations[ann["image_id"]].append(ann)
        total_annotations += 1

    print(f"有效标注总数：{total_annotations}，跳过：{total_skipped}")

    # ——————————————————————————————————————————————
    # 4. 逐图像生成 YOLO 标签文件
    # ——————————————————————————————————————————————
    processed_images = 0
    written_labels = 0

    for img_id, annotations in img_annotations.items():
        if img_id not in id_to_image:
            continue

        img_info = id_to_image[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        if img_w <= 0 or img_h <= 0:
            continue

        # 获取图像文件名（去掉扩展名作为标签文件名）
        img_filename = Path(img_info["file_name"]).stem
        label_file = output_path / f"{img_filename}.txt"

        lines = []
        for ann in annotations:
            x_min, y_min, bbox_w, bbox_h = ann["bbox"]

            # 边界框转为中心坐标并归一化
            cx = (x_min + bbox_w / 2.0) / img_w
            cy = (y_min + bbox_h / 2.0) / img_h
            nw = bbox_w / img_w
            nh = bbox_h / img_h

            # 跳过归一化后面积过小的实例
            if nw * nh < min_area:
                continue

            # 将所有值限制到 [0, 1]（避免因标注越界导致的异常）
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            # 处理关键点（COCO 格式：每个关键点 3 个值 x, y, visibility）
            raw_kpts = ann.get("keypoints", [])

            # 若没有关键点字段或数据不完整，用全零填充至 17 个关键点
            expected_len = COCO_NUM_KEYPOINTS * 3
            if len(raw_kpts) < expected_len:
                if len(raw_kpts) > 0:
                    # 部分关键点数据，可能是标注不完整，打印警告
                    print(
                        f"  [警告] 图像 {img_id} 标注 {ann['id']} 的关键点数据不完整："
                        f"期望 {expected_len} 个值，实际 {len(raw_kpts)} 个，已用零值填充。"
                    )
                raw_kpts = raw_kpts + [0] * (expected_len - len(raw_kpts))

            kpt_values = []
            for k in range(COCO_NUM_KEYPOINTS):
                kx = raw_kpts[k * 3]       # 关键点像素 x 坐标
                ky = raw_kpts[k * 3 + 1]   # 关键点像素 y 坐标
                kv = int(raw_kpts[k * 3 + 2])  # 可见性（0/1/2）

                # 坐标归一化
                kx_norm = kx / img_w if kv > 0 else 0.0
                ky_norm = ky / img_h if kv > 0 else 0.0

                # 限制到 [0, 1]
                kx_norm = max(0.0, min(1.0, kx_norm))
                ky_norm = max(0.0, min(1.0, ky_norm))

                kpt_values.extend([kx_norm, ky_norm, kv])

            # 构建 YOLO 格式的标签行：class_id cx cy w h kx1 ky1 kv1 ...
            # 人的类别索引为 0
            values = [0, cx, cy, nw, nh] + kpt_values
            line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)
            lines.append(line)

        # 写入标签文件（即使没有有效标注也创建空文件，方便数据加载器处理）
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        processed_images += 1
        written_labels += len(lines)

    print(f"处理完成：共处理 {processed_images} 张图像，写入 {written_labels} 条标签")
    print(f"标签文件保存目录：{output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将 COCO train2017/val2017 人体关键点 JSON 标注转换为 YOLO 格式标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 转换训练集标注
  python utils/coco2yolo_keypoints.py \\
      --json datasets/coco-pose/annotations/person_keypoints_train2017.json \\
      --output datasets/coco-pose/labels/train2017

  # 转换验证集，跳过没有关键点的实例
  python utils/coco2yolo_keypoints.py \\
      --json datasets/coco-pose/annotations/person_keypoints_val2017.json \\
      --output datasets/coco-pose/labels/val2017 \\
      --require-keypoints
        """,
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="COCO 关键点标注 JSON 文件路径（如 person_keypoints_train2017.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 YOLO 标签 .txt 文件的目录",
    )
    parser.add_argument(
        "--require-keypoints",
        action="store_true",
        default=False,
        help="跳过 num_keypoints=0 的实例（没有任何可见关键点的实例）",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="最小归一化边界框面积阈值（默认 0.0，不过滤）",
    )
    parser.add_argument(
        "--person-category-id",
        type=int,
        default=None,
        help="人体类别在 COCO JSON 中的 category_id（默认自动检测）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_coco_to_yolo_keypoints(
        json_path=args.json,
        output_dir=args.output,
        require_keypoints=args.require_keypoints,
        min_area=args.min_area,
        person_category_id=args.person_category_id,
    )
