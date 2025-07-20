import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image

import config
from utils import iou_width_height as iou, non_max_suppression as nms


class YOLODatasetTF(tf.data.Dataset):
    def __new__(
        cls,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        df = pd.read_csv(csv_file)
        img_paths = [os.path.join(img_dir, x) for x in df.iloc[:, 0]]
        label_paths = [os.path.join(label_dir, x) for x in df.iloc[:, 1]]
        anchors = np.array(anchors[0] + anchors[1] + anchors[2], dtype=np.float32)

        num_anchors = anchors.shape[0]
        num_anchors_per_scale = num_anchors // 3

        def parse_example(img_path, label_path):
            img_path = img_path.numpy().decode()
            label_path = label_path.numpy().decode()

            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            bboxes = np.loadtxt(label_path, delimiter=" ", ndmin=2)
            bboxes = np.roll(bboxes, 4, axis=1).tolist()  # convert to xywhc

            if transform:
                augmented = transform(image=image, bboxes=bboxes)
                image = augmented["image"].numpy()
                bboxes = augmented["bboxes"]

            targets = [
                np.zeros((num_anchors_per_scale, S_i, S_i, 6), dtype=np.float32)
                for S_i in S
            ]

            for box in bboxes:
                x, y, width, height, class_label = box
                iou_anchors = iou(np.array([width, height]), anchors)
                anchor_indices = np.argsort(-iou_anchors)

                has_anchor = [False, False, False]

                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // num_anchors_per_scale
                    anchor_on_scale = anchor_idx % num_anchors_per_scale
                    S_i = S[scale_idx]

                    i, j = int(S_i * y), int(S_i * x)
                    if i >= S_i or j >= S_i:
                        continue

                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S_i * x - j, S_i * y - i
                        width_cell, height_cell = width * S_i, height * S_i
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = [
                            x_cell,
                            y_cell,
                            width_cell,
                            height_cell,
                        ]
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True

                    elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1

            return image.astype(np.float32), tuple(targets)

        def tf_wrapper(img_path, label_path):
            image, targets = tf.py_function(
                func=parse_example,
                inp=[img_path, label_path],
                Tout=(tf.float32, (tf.float32, tf.float32, tf.float32)),
            )
            image.set_shape([config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
            for i, S_i in enumerate(S):
                targets[i].set_shape([num_anchors_per_scale, S_i, S_i, 6])
            return image, targets

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
        dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        return dataset
