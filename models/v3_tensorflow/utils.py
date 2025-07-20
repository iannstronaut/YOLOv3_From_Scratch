import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random, os
from collections import Counter


def iou_width_height(boxes1, boxes2):
    intersect = tf.minimum(boxes1[..., 0], boxes2[..., 0]) * tf.minimum(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersect
    )
    return intersect / union


def intersection_over_union(box1, box2, box_format="midpoint"):
    if box_format == "midpoint":
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2
    else:  # corners
        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(box2, 4, axis=-1)

    x1 = tf.maximum(b1_x1, b2_x1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x2 = tf.minimum(b1_x2, b2_x2)
    y2 = tf.minimum(b1_y2, b2_y2)
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    area1 = tf.abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    area2 = tf.abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    return inter / (area1 + area2 - inter + 1e-6)


def non_max_suppression_tf(bboxes, iou_thresh, score_thresh, box_format="corners"):
    # bboxes: numpy array [[cls,score,x1,y1,x2,y2],...]
    bboxes = [box for box in bboxes if box[1] > score_thresh]
    if not bboxes:
        return []
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    keep = []
    while bboxes:
        chosen = bboxes.pop(0)
        keep.append(chosen)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen[0]
            or intersection_over_union(
                tf.convert_to_tensor(chosen[2:], dtype=tf.float32),
                tf.convert_to_tensor(box[2:], dtype=tf.float32),
                box_format=box_format,
            )
            < iou_thresh
        ]
    return keep


def mean_average_precision_tf(
    pred_boxes, true_boxes, iou_thresh=0.5, box_format="midpoint", num_classes=20
):
    average_precisions = []
    eps = 1e-6
    for c in range(num_classes):
        detections = [b for b in pred_boxes if b[1] == c]
        ground_truths = [b for b in true_boxes if b[1] == c]
        gt_counter = Counter([gt[0] for gt in ground_truths])
        gt_flags = {
            idx: tf.zeros(gt_counter[idx], dtype=tf.int32) for idx in gt_counter
        }
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = tf.zeros(len(detections))
        FP = tf.zeros(len(detections))
        total_gt = len(ground_truths)
        if total_gt == 0:
            continue
        for idx, det in enumerate(detections):
            img_gt = [gt for gt in ground_truths if gt[0] == det[0]]
            best_iou = 0
            best_idx = None
            for gi, gt in enumerate(img_gt):
                iou = intersection_over_union(
                    tf.convert_to_tensor(det[3:], tf.float32),
                    tf.convert_to_tensor(gt[3:], tf.float32),
                    box_format,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou > iou_thresh and gt_flags[det[0]][best_idx] == 0:
                TP = tf.tensor_scatter_nd_update(TP, [[idx]], [1.0])
                gt_flags[det[0]] = tf.tensor_scatter_nd_update(
                    gt_flags[det[0]], [[best_idx]], [1]
                )
            else:
                FP = tf.tensor_scatter_nd_update(FP, [[idx]], [1.0])
        cum_TP = tf.cumsum(TP)
        cum_FP = tf.cumsum(FP)
        recalls = cum_TP / (total_gt + eps)
        precisions = cum_TP / (cum_TP + cum_FP + eps)
        precisions = tf.concat([[1.0], precisions], axis=0)
        recalls = tf.concat([[0.0], recalls], axis=0)
        average_precisions.append(
            tf.reduce_sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        )
    return tf.reduce_mean(average_precisions) if average_precisions else 0


def plot_image(image, boxes, class_labels):
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    h, w = image.shape[:2]
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        cls, conf, x, y, bw, bh = box
        ulx = x - bw / 2
        uly = y - bh / 2
        rect = patches.Rectangle(
            (ulx * w, uly * h),
            bw * w,
            bh * h,
            linewidth=2,
            edgecolor=colors[int(cls)],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            ulx * w,
            uly * h,
            class_labels[int(cls)],
            color="white",
            bbox={"facecolor": colors[int(cls)], "pad": 0},
        )
    plt.show()


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
