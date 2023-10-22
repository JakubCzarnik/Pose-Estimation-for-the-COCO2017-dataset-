import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Concatenate, UpSampling2D, Add, Activation




def residual_module(inputs, n_filters):
    merge_input = inputs
    if inputs.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Activation('relu')(conv2)

    output = Add()([conv2, merge_input])
    output = Activation('relu')(output)
    return output


def upsample_concat(input_A, input_B):
    upsample = UpSampling2D((2, 2))(input_A)
    concat = Concatenate()([upsample, input_B])
    return concat


def build_model(heat_filters, paf_filters, k=32, input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    x = inputs
    for _ in range(2):
        x = residual_module(x, k)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(4):
        x = residual_module(x, 2*k)
   
    x0 = x
    x = MaxPooling2D((2, 2))(x)
    for _ in range(4):
        x = residual_module(x, 4*k)

    x1 = x 
    x = MaxPooling2D((2, 2))(x)
    for _ in range(6):
        x = residual_module(x, 8*k)
    
    x2 = x
    x = MaxPooling2D((2, 2))(x)
    for _ in range(7):
        x = residual_module(x, 8*k)
    x3 = x

    y1 = residual_module(x1, 4*k) # 1/4
    y2 = residual_module(x2, 8*k) # 1/8
    y3 = residual_module(x3, 8*k) # 1/16

    # Concatenate 
    y21 = upsample_concat(y2, y1) # 1/8 & 1/4
    y21 = residual_module(y21, 4*k) # 1/4

    y32 = upsample_concat(y3, y2) # 1/16 & 1/8
    y32 = residual_module(y32, 8*k) # 1/8

    y32_21 = upsample_concat(y32, y21) # 1/8 & 1/4
    y32_21 = residual_module(y32_21, 8*k) # 1/4

    ### Confidence maps ###
    heat =  y32_21
    for i in range(8):
        if i != 0:
            heat = BatchNormalization()(heat)
        heat = Conv2D(4*k, kernel_size=7, padding="Same", kernel_initializer='he_normal')(heat)
        heat = Activation('relu')(heat)

    heat_0 = heat
    heat = Conv2D(heat_filters, kernel_size=1, activation="sigmoid", padding="Same", kernel_initializer='he_normal', name="heat_out")(heat)
    

    ### PAFs ###
    pafs = upsample_concat(y32, y21) # 1/8 & 1/4
    pafs = residual_module(pafs, 8*k) # 1/4
    for i in range(8):
        if i != 0:
            pafs = BatchNormalization()(pafs)
        pafs = Conv2D(4*k, kernel_size=7, padding="Same", kernel_initializer='he_normal')(pafs)
        pafs = Activation('relu')(pafs)

    pafs = Concatenate()([pafs, heat_0])
    pafs = Conv2D(paf_filters, kernel_size=1, activation="tanh", padding="Same", kernel_initializer='he_normal', name="paf_out")(pafs)


    model = tf.keras.Model(inputs=inputs, outputs=[heat, pafs])
    return model


if __name__ == "__main__":
    model_hrnet = build_model(18, 34)
    model_hrnet.summary()


