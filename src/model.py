import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Concatenate, UpSampling2D, Add, Activation




def residual_module(inputs, n_filters, momentum=0.9):
    merge_input = inputs
    if inputs.shape[-1] != n_filters:
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
        merge_input = BatchNormalization(momentum=momentum)(merge_input)
        merge_input = Activation('relu')(merge_input)

    conv1 = Conv2D(n_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,)(inputs)
    conv1 = BatchNormalization(momentum=momentum)(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(n_filters, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization(momentum=momentum)(conv2)

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
    for _ in range(8):
        x = residual_module(x, 4*k)

    x1 = x 
    x = MaxPooling2D((2, 2))(x)
    for _ in range(6):
        x = residual_module(x, 8*k)
    
    x2 = x
    x = MaxPooling2D((2, 2))(x)
    for _ in range(4):
        x = residual_module(x, 8*k)
    x3 = x

    y1 = residual_module(x1, 4*k) # 1/4
    y2 = residual_module(x2, 8*k) # 1/8
    y3 = residual_module(x3, 8*k) # 1/16

    # Concatenate 
    y21 = upsample_concat(y2, y1) # 1/8 & 1/4
    for _ in range(2):
        y21 = residual_module(y21, 4*k) # 1/4

    y32 = upsample_concat(y3, y2) # 1/16 & 1/8
    for _ in range(2):
        y32 = residual_module(y32, 8*k) # 1/8

    ### Confidence maps ###
    y32_21_heat = upsample_concat(y32, y21) # 1/8 & 1/4
    y32_21_heat = residual_module(y32_21_heat, 16*k) # 1/4

    heat =  y32_21_heat
    for _ in range(5):
        heat = residual_module(heat, 8*k)
        
    heat_0 = heat
    heat = Conv2D(heat_filters, kernel_size=3, activation="sigmoid", padding="Same", kernel_initializer='he_normal', name="heat_out")(heat)
    

    ### PAFs ###
    y32_21_pafs = upsample_concat(y32, y21) # 1/8 & 1/4
    y32_21_pafs = residual_module(y32_21_pafs, 16*k) # 1/4

    pafs =  y32_21_heat
    for _ in range(5):
        pafs = residual_module(pafs, 8*k)

    pafs = Concatenate()([pafs, heat_0])
    pafs = Conv2D(paf_filters, kernel_size=3, activation="tanh", padding="Same", kernel_initializer='he_normal', name="paf_out")(pafs)


    model = tf.keras.Model(inputs=inputs, outputs=[heat, pafs])
    return model


if __name__ == "__main__":
    model_hrnet = build_model(18, 34)
    model_hrnet.summary()


