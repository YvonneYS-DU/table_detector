"""
Simple expand & merge utility for table detection bounding boxes.

This module provides a small, independent component that accepts the output
from `wire_detector.detect_tables(...)` (a list of dicts with a 'bbox' key
containing [x1, y1, x2, y2]) and performs a simple expansion and merging pass.

Usage:
    from merge_expand import expand_and_merge_tables

    merged = expand_and_merge_tables(tables, expand_px=5, iou_threshold=0.3)

Also provides a CLI to load a JSON file (either a list of table dicts or a dict
with a 'tables' key), run the merge, write output JSON and optionally create a
visualization image showing original and merged boxes.
"""

from typing import List, Tuple, Optional, Dict, Any
import json
import cv2
import numpy as np
import os

BBox = Tuple[float, float, float, float]


def _bbox_area(b: BBox) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersect(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    """Compute IoU between two bboxes."""
    inter = _intersect(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def expand_bbox(b: BBox, expand_px: float, image_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """
    Expand bbox by a fixed number of pixels (can be float). If expand_px is
    between 0 and 1, it will be treated as fraction of bbox width/height and
    applied proportionally.
    """
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    if 0.0 < expand_px < 1.0:
        dx = w * expand_px
        dy = h * expand_px
    else:
        dx = expand_px
        dy = expand_px

    nx1 = x1 - dx
    ny1 = y1 - dy
    nx2 = x2 + dx
    ny2 = y2 + dy

    if image_shape is not None:
        ih, iw = image_shape[:2]
        nx1 = max(0.0, nx1)
        ny1 = max(0.0, ny1)
        nx2 = min(float(iw - 1), nx2)
        ny2 = min(float(ih - 1), ny2)

    return (nx1, ny1, nx2, ny2)


def union_bbox(boxes: List[BBox]) -> BBox:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def merge_bboxes_greedy(bboxes: List[BBox], iou_threshold: float) -> List[BBox]:
    """
    Greedy merging: repeatedly take the first box and merge it with any other
    boxes whose IoU exceeds iou_threshold (union). Simpler than hierarchical
    clustering but good enough for small numbers of boxes.
    """
    if not bboxes:
        return []

    boxes = bboxes.copy()
    merged: List[BBox] = []

    while boxes:
        base = boxes.pop(0)
        to_merge = [base]
        i = 0
        # iterate over remaining boxes and collect those with IoU > threshold
        while i < len(boxes):
            if iou(base, boxes[i]) > iou_threshold:
                to_merge.append(boxes.pop(i))
            else:
                i += 1

        # after merging some, new union may overlap others; try to absorb more
        merged_box = union_bbox(to_merge)
        changed = True
        while changed:
            changed = False
            j = 0
            while j < len(boxes):
                if iou(merged_box, boxes[j]) > iou_threshold:
                    to_merge.append(boxes.pop(j))
                    merged_box = union_bbox(to_merge)
                    changed = True
                else:
                    j += 1

        merged.append(merged_box)

    return merged


def expand_and_merge_tables(
    tables: List[Dict[str, Any]],
    expand_px: float = 5.0,
    iou_threshold: float = 0.3,
    image_shape: Optional[Tuple[int, int]] = None,
    keep_original: bool = False,
) -> List[Dict[str, Any]]:
    """
    Expand and merge simple rectangles from `detect_tables` output.

    Inputs:
      - tables: list of dicts, each dict must have a 'bbox' key with [x1,y1,x2,y2]
      - expand_px: pixels to expand each box (or fraction if between 0 and 1)
      - iou_threshold: IoU threshold above which boxes will be merged
      - image_shape: optional (height, width) to clamp expanded boxes
      - keep_original: if True, keep original table dicts in 'merged_from' mapping

    Returns: list of dicts with keys:
      - 'bbox': merged bbox [x1,y1,x2,y2]
      - 'merged_from': list of indices (into original tables) merged into this
    """
    # extract bboxes
    orig_bboxes: List[BBox] = []
    for t in tables:
        b = t.get('bbox')
        if b is None:
            continue
        # ensure floats
        if len(b) >= 4:
            orig_bboxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))

    expanded = [expand_bbox(b, expand_px, image_shape=image_shape) for b in orig_bboxes]

    merged = merge_bboxes_greedy(expanded, iou_threshold=iou_threshold)

    # build mapping back to original indices by checking intersection
    out: List[Dict[str, Any]] = []
    for m in merged:
        merged_from = []
        for idx, e in enumerate(expanded):
            # consider overlap ratio over original area
            inter = _intersect(m, e)
            if _bbox_area(e) <= 0:
                continue
            if inter / _bbox_area(e) > 0.01:  # any touching/overlap
                merged_from.append(idx)

        # fall back: if none matched, try IoU mapping
        if not merged_from:
            for idx, e in enumerate(expanded):
                if iou(m, e) > 0.0:
                    merged_from.append(idx)

        out.append({
            'bbox': [float(m[0]), float(m[1]), float(m[2]), float(m[3])],
            'merged_from': merged_from,
        })

    # Optionally include original entries
    if keep_original:
        for i, t in enumerate(tables):
            t['_orig_index'] = i

    return out


def visualize_merge(
    image_path: str,
    original_tables: List[Dict[str, Any]],
    merged_tables: List[Dict[str, Any]],
    out_path: str,
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    vis = img.copy()

    # draw originals (blue)
    for t in original_tables:
        b = t.get('bbox')
        if not b:
            continue
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # draw merged (red, thicker)
    for m in merged_tables:
        b = m.get('bbox')
        if not b:
            continue
        x1, y1, x2, y2 = map(int, b[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # annotate how many were merged
        cnt = len(m.get('merged_from', []))
        cv2.putText(vis, str(cnt), (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(out_path, vis)
    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Expand and merge detected table boxes')
    parser.add_argument('--input', '-i', default='page_0_tables.json', help='Input JSON with tables or {"tables": [...]}')
    parser.add_argument('--image', '-m', default=None, help='Optional image path for visualization')
    parser.add_argument('--expand', '-e', type=float, default=5.0, help='Pixels (or fraction if 0<e<1) to expand each bbox')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold for merging')
    parser.add_argument('--out', '-o', default=None, help='Output JSON path')
    parser.add_argument('--vis', action='store_true', help='Save visualization image (requires --image)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    with open(args.input, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    if isinstance(data, dict) and 'tables' in data:
        tables = data['tables']
    elif isinstance(data, list):
        tables = data
    else:
        raise SystemExit('Input JSON must be a list of table dicts or a dict with key "tables"')

    merged = expand_and_merge_tables(tables, expand_px=args.expand, iou_threshold=args.iou)

    out_path = args.out or (os.path.splitext(args.input)[0] + '_merged.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump({'tables': merged}, fp, ensure_ascii=False, indent=2)

    print(f'Wrote merged results: {out_path} (original: {len(tables)} -> merged: {len(merged)})')

    if args.vis:
        if not args.image:
            raise SystemExit('Visualization requested but no --image provided')
        vis_out = os.path.splitext(out_path)[0] + '_merged_vis.jpg'
        visualize_merge(args.image, tables, merged, vis_out)
        print(f'Wrote visualization: {vis_out}')
