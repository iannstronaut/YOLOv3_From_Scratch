import config
import tensorflow as tf
from model import YOLOv3  # Pastikan ini versi TensorFlow
from loss import YOLOLoss  # Versi TensorFlow dari YOLOLoss
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    load_checkpoint,
    save_checkpoint,
)


def train_fn(train_dataset, model, optimizer, loss_fn, scaled_anchors):
    total_loss = 0
    num_batches = 0

    for batch, (images, targets) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            loss = (
                loss_fn(outputs[0], targets[0], scaled_anchors[0])
                + loss_fn(outputs[1], targets[1], scaled_anchors[1])
                + loss_fn(outputs[2], targets[2], scaled_anchors[2])
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss.numpy()
        num_batches += 1
        print(f"Batch {batch+1}: Loss = {loss.numpy():.4f}")

    mean_loss = total_loss / num_batches
    print(f"Epoch training loss: {mean_loss:.4f}")


def main():
    # Setup model, optimizer, and loss
    strategy = tf.distribute.get_strategy()  # For multi-GPU compatibility
    with strategy.scope():
        model = YOLOv3(num_classes=config.NUM_CLASSES)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
        )
        loss_fn = YOLOLoss()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/100examples.csv",
        test_csv_path=config.DATASET + "/100examples.csv",
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer)

    # Compute scaled anchors
    anchors = tf.convert_to_tensor(config.ANCHORS, dtype=tf.float32)
    S = tf.convert_to_tensor(config.S, dtype=tf.float32)
    scaled_anchors = anchors * tf.reshape(S, (3, 1, 1))

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename="checkpoint.tf")

        if epoch % 10 == 0 and epoch > 0:
            print("Evaluation on test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"mAP: {mapval:.4f}")


if __name__ == "__main__":
    main()
