import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, concatenate, Activation, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal


class Conv(Conv2D):
    def __init__(self, filters, kernel, strides=(1,1), activation='relu', padding="same", use_bn=False, **kwargs):
        super().__init__(filters, kernel, strides=strides, kernel_initializer=GlorotNormal(), padding=padding, **kwargs)
        self.activation = Activation(activation)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = BatchNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_bn": self.use_bn,
            "activation": self.activation,
        })
        return config


    def call(self, inputs):
        x = super().call(inputs)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


def conv_block(inputs, filters, kernel, convs_times=3, use_bn=False):
    x = inputs
    for _ in range(convs_times):
        x = Conv(filters, kernel, use_bn=use_bn)(x)
    return x


def map_block(inputs, filters, heat_filters, paf_filters, kernel, convs_times=1, use_bn=False):
    # heatmaps
    x1 = inputs
    for _ in range(convs_times):
        x1 = Conv(filters, kernel, use_bn=use_bn)(x1)
    x1 = Conv(heat_filters, kernel=1, use_bn=use_bn)(x1)
    # pafs
    x2 = inputs
    for _ in range(convs_times):
        x2 = Conv(filters, kernel, use_bn=use_bn)(x2)
    x2 = Conv(paf_filters, kernel=1, use_bn=use_bn)(x2)
    return x1, x2


def build_model(heat_filters, paf_filters):
    i = Input((386,386,3))

    x = conv_block(i, 64, kernel=3, convs_times=3)
    x = MaxPool2D((2,2))(x)
    x = conv_block(x, 80, kernel=3, convs_times=3)
    x = MaxPool2D((2,2))(x)
    x = conv_block(x, 96, kernel=3, convs_times=4)
    x = MaxPool2D((2,2))(x)
    x = conv_block(x, 128, kernel=3, convs_times=6)


    heat1, paf1 = map_block(x, 96, heat_filters, paf_filters, kernel=5, convs_times=4)
    x = concatenate([x, heat1, paf1])

    heat2, paf2 = map_block(x, 96, heat_filters, paf_filters, kernel=7, convs_times=6)
    x = concatenate([x, heat2, paf2])

    heat3, paf3 = map_block(x, 96, heat_filters, paf_filters, kernel=7, convs_times=4)
    x = concatenate([x, heat3, paf3])

    heat4, paf4 = map_block(x, 96, heat_filters, paf_filters, kernel=7, convs_times=6)
    x = concatenate([x, heat4, paf4])

    heat5, paf5 = map_block(x, 96, heat_filters, paf_filters, kernel=7, convs_times=4)
    x = concatenate([x, heat5, paf5])


    x1 = conv_block(x, 96, kernel=7, convs_times=7)
    heat = Conv(heat_filters, kernel=1, activation="sigmoid")(x1)

    x2 = conv_block(x, 96, kernel=7, convs_times=7)
    paf = Conv(paf_filters, kernel=1, activation="tanh")(x2)

    model = tf.keras.models.Model(i, [heat, paf])
    return model

if __name__ == "__main__":
    model = build_model(18, 34)
    model.summary()