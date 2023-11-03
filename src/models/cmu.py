import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Input, LeakyReLU, MaxPool2D
from models.builder import ModelConstructor

class CMU(ModelConstructor):
   def __init__(self, config):
      super().__init__(config)
      

   def build_model(self):
      i = Input(self.input_shape)
      x = i
      for _ in range(2):
         x = self.conv(x, 64, (3,3))
      x = MaxPool2D()(x)
      for _ in range(2):
         x = self.conv(x, 128, (3,3))
      x = MaxPool2D()(x)
      for _ in range(4):
         x = self.conv(x, 256, (3,3))
      x = MaxPool2D()(x)
      for _ in range(2):
         x = self.conv(x, 512, (3,3))
      x = self.conv(x, 256, (3,3))
      x = self.conv(x, 128, (3,3))

      y0 = x
      heat0, paf0 = self.split(y0, 3, 128, (3,3), 512, (1,1), self.heat_filters, self.paf_filters)
      
      y1 = Concatenate()([paf0, heat0, y0])
      heat1, paf1 = self.split(y1, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters)

      y2 = Concatenate()([paf1, heat1, y0])
      heat2, paf2 = self.split(y2, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters)

      y3 = Concatenate()([paf2, heat2, y0])
      heat3, paf3 = self.split(y3, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters)

      y4 = Concatenate()([paf3, heat3, y0])
      heat4, paf4 = self.split(y4, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters)

      y5 = Concatenate()([paf4, heat4, y0])
      heat5, paf5 = self.split(y5, 5, 128, (7,7), 128, (1,1), self.heat_filters, self.paf_filters, last=True)

      model = tf.keras.models.Model(inputs=i, outputs=[heat5, paf5])
      return model


if __name__ == "__main__":
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
   from config import Config

   cfg = Config()
   mc = CMU(cfg)
   model = mc.build_model() 
   model.summary()