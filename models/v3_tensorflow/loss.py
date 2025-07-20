import tensorflow as tf

# Ganti fungsi IoU sesuai dengan implementasimu
from utils import intersection_over_union  # Pastikan ini versi TF


class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def call(self, y_pred, y_true, anchors):
        obj_mask = tf.equal(y_true[..., 0], 1)
        noobj_mask = tf.equal(y_true[..., 0], 0)

        if tf.reduce_sum(tf.cast(obj_mask, tf.int32)) == 0:
            tf.print(
                "Tidak ada object yang diassign ke anchor. Cek ukuran anchor vs box."
            )

        # No object loss
        no_object_loss = self.bce(
            y_true[..., 0:1][noobj_mask], y_pred[..., 0:1][noobj_mask]
        )

        # Object loss
        anchors = tf.reshape(anchors, (1, 3, 1, 1, 2))
        pred_box_xy = tf.sigmoid(y_pred[..., 1:3])
        pred_box_wh = tf.exp(y_pred[..., 3:5]) * anchors
        box_preds = tf.concat([pred_box_xy, pred_box_wh], axis=-1)

        ious = intersection_over_union(box_preds[obj_mask], y_true[..., 1:5][obj_mask])
        ious = tf.stop_gradient(ious)
        object_loss = self.bce(
            y_true[..., 0:1][obj_mask] * ious, y_pred[..., 0:1][obj_mask]
        )

        # Box Coordinate loss
        pred_box = tf.identity(y_pred)
        pred_box = tf.tensor_scatter_nd_update(
            pred_box,
            tf.where(obj_mask),
            tf.concat(
                [tf.sigmoid(y_pred[..., 1:3])[obj_mask], y_pred[..., 3:5][obj_mask]],
                axis=-1,
            ),
        )

        y_true_adj = tf.identity(y_true)
        true_wh = y_true[..., 3:5] / (anchors + 1e-16)
        true_wh = tf.math.log(tf.clip_by_value(true_wh, 1e-9, 1e9))
        y_true_adj = tf.tensor_scatter_nd_update(
            y_true_adj,
            tf.where(obj_mask),
            tf.concat([y_true[..., 1:3][obj_mask], true_wh[obj_mask]], axis=-1),
        )

        box_loss = self.mse(
            pred_box[..., 1:5][obj_mask], y_true_adj[..., 1:5][obj_mask]
        )

        # Class loss
        class_loss = self.entropy(y_true[..., 5][obj_mask], y_pred[..., 5:][obj_mask])

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
