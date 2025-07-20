import tensorflow as tf
from tensorflow.keras import layers, Model

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # Sampai sini adalah Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bn=True):
        super().__init__()
        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding="same" if kernel_size == 3 else "valid",
            use_bias=not use_bn,
        )
        self.bn = layers.BatchNormalization() if use_bn else None
        self.activation = layers.LeakyReLU(alpha=0.1) if use_bn else None

    def call(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(layers.Layer):
    def __init__(self, filters, repeats=1, use_residual=True):
        super().__init__()
        self.blocks = []
        for _ in range(repeats):
            self.blocks.append([CNNBlock(filters // 2, 1, 1), CNNBlock(filters, 3, 1)])
        self.use_residual = use_residual

    def call(self, x):
        for conv1, conv2 in self.blocks:
            residual = x
            x = conv1(x)
            x = conv2(x)
            if self.use_residual:
                x += residual
        return x


class ScalePrediction(layers.Layer):
    def __init__(self, filters, num_classes):
        super().__init__()
        self.conv1 = CNNBlock(filters * 2, 3, 1)
        self.conv2 = CNNBlock(3 * (num_classes + 5), 1, 1, use_bn=False)
        self.num_classes = num_classes

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        batch_size = tf.shape(x)[0]
        grid_size = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, grid_size, grid_size, 3, self.num_classes + 5))
        return x


class YOLOv3(Model):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.layers_list = []
        self._build_layers()

    def _build_layers(self):
        in_filters = 3
        self.route_connections = []

        for module in config:
            if isinstance(module, tuple):
                filters, kernel_size, stride = module
                self.layers_list.append(CNNBlock(filters, kernel_size, stride))
                in_filters = filters

            elif isinstance(module, list):
                _, repeats = module
                self.layers_list.append(ResidualBlock(in_filters, repeats))

            elif isinstance(module, str):
                if module == "S":
                    self.layers_list.append(
                        ResidualBlock(in_filters, 1, use_residual=False)
                    )
                    self.layers_list.append(CNNBlock(in_filters // 2, 1, 1))
                    self.layers_list.append(
                        ScalePrediction(in_filters // 2, self.num_classes)
                    )
                    in_filters = in_filters // 2

                elif module == "U":
                    self.layers_list.append(layers.UpSampling2D(2))
                    in_filters = in_filters * 3

    def call(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers_list:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
            else:
                x = layer(x)

                if (
                    isinstance(layer, ResidualBlock)
                    and layer.blocks
                    and len(layer.blocks) == 8
                ):
                    route_connections.append(x)

                if isinstance(layer, layers.UpSampling2D):
                    skip_connection = route_connections.pop()
                    x = tf.concat([skip_connection, x], axis=-1)

        return outputs


# Testing
if __name__ == "__main__":
    IMAGE_SIZE = 416
    num_classes = 20
    model = YOLOv3(num_classes)
    dummy_input = tf.random.normal((2, IMAGE_SIZE, IMAGE_SIZE, 3))
    outputs = model(dummy_input)

    assert outputs[0].shape == (
        2,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        3,
        num_classes + 5,
    )
    assert outputs[1].shape == (
        2,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        3,
        num_classes + 5,
    )
    assert outputs[2].shape == (2, IMAGE_SIZE // 8, IMAGE_SIZE // 8, 3, num_classes + 5)
    print("Success!")
