from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, LeakyReLU, UpSampling2D
import tensorflow as tf

class ModelConstructor:
   def __init__(self, config):
      self.input_shape = (*config.target_image_size, 3)
      self.heat_filters = config.n_keypoints
      self.paf_filters = 2*len(config.pairs)
      self._l1 = config.l1
      self._l2 = config.l2
      self.bn_momentum = config.bn_momentum
      self.leaky_alpha = config.leaky_alpha
      

   def conv(self, inputs, filters, kernel_size, act=True, bn=True, **kwargs):
      config = {"padding": "same",
                "kernel_initializer": "he_normal"}
      config.update(kwargs)

      x = Conv2D(filters, kernel_size, **config)(inputs)
      if bn:
         x = BatchNormalization(momentum=self.bn_momentum)(x)
      if act and "activation" not in config:
         x = LeakyReLU(self.leaky_alpha)(x)
      return x


   def split(self, inputs, n_convs, filters1, kernel1, filters2, kernel2, heat_filters, paf_filters, last=False):
      x = inputs ### heat
      for _ in range(n_convs):
         x = self.conv(x, filters1, kernel1, kernel_regularizer=l2(self._l2))
      x = self.conv(x, filters2, kernel2)

      if last:
         heat = self.conv(x, heat_filters, kernel_size=1, activation="sigmoid", bn=False,
                          name="heat_out", kernel_regularizer=l2(self._l2))
      else:
         heat = self.conv(x, heat_filters, (1,1), act=False, bn=False)
      
      x = inputs ### pafs
      for _ in range(n_convs):
         x = self.conv(x, filters1, kernel1)
      x = self.conv(x, filters2, kernel2, kernel_regularizer=l1(self._l1))

      if last:
         paf = self.conv(x, paf_filters, kernel_size=1, activation="tanh", bn=False,
                          name="paf_out", kernel_regularizer=l1(self._l1))
      else:
         paf = self.conv(x, paf_filters, (1,1), act=False, bn=False)
      return heat, paf
   

   def upsample_concat(self, input_A, input_B):
      upsample = UpSampling2D((2, 2))(input_A)
      concat = Concatenate()([upsample, input_B])
      return concat
   

   def spatial_attention(self, inputs, filters1, filters2, kernel_size=(7,7)):
      avg_out = tf.reduce_mean(inputs, axis=3)
      max_out = tf.reduce_max(inputs, axis=3)
      x = tf.stack([avg_out, max_out], axis=3)
      x = self.conv(x, filters1, kernel_size=kernel_size, bn=False)
      x = self.conv(x, filters2, kernel_size=kernel_size, bn=False, activation='sigmoid')
      return x * inputs
