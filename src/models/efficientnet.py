import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Concatenate, Conv2D
from keras_efficientnet_v2 import EfficientNetV2B0
from models.builder import ModelConstructor


class EfficientNet(ModelConstructor):
   def __init__(self, config):
      super().__init__(config)

   def get_base_model(self):
      base_model = EfficientNetV2B0(input_shape=self.input_shape)
      #base_model.summary() # 7.2 kk
      
      layers = ["post_swish", "stack_5_block0_sortcut_swish", "stack_3_block0_sortcut_swish", "add"]
      #1/32  1/16  1/8  1/4
      out1, out2, out3, out4 = [base_model.get_layer(layer).output for layer in layers]
      base_model.trainable = False

      y1 = self.upsample_concat(out1, out2) # 1/32 & 1/16 --> 1/16
      y1 = self.conv(y1, 384, kernel_size=(3,3), kernel_regularizer=l2(self._l2))
      
      y2 = self.upsample_concat(y1, out3)   # 1/16 & 1/8  --> 1/8
      y2 = self.conv(y2, 256, kernel_size=(5,5),  kernel_regularizer=l2(self._l2))

      y3 = self.upsample_concat(y2, out4)   # 1/8 & 1/4  --> 1/4
      y3 = self.conv(y3, 128, kernel_size=(7,7),  kernel_regularizer=l2(self._l2))

      model = tf.keras.models.Model(inputs=base_model.input, outputs=y3, name="EfficientNetV2B0")
      #model.summary()
      return model


   def build_model(self):
      base_model = self.get_base_model()
      
      i = Input(self.input_shape)
      y0 = base_model(i)


      heat0, paf0 = self.split(y0, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters)
      heat0 = self.spatial_attention(heat0, 64, self.heat_filters, kernel_size=(7,7))
      paf0 = self.spatial_attention(paf0, 64, self.paf_filters, kernel_size=(7,7))


      y1 = Concatenate()([paf0, heat0, y0])
      heat1, paf1 = self.split(y1, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters, last=True)

      model = tf.keras.models.Model(inputs=i, outputs=[heat1, paf1])
      return model



if __name__ == "__main__":
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
   from config import Config

   cfg = Config()

   model = EfficientNet(cfg).build_model()
   model.summary()


