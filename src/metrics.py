import tensorflow as tf



class HeatLoss(tf.keras.losses.Loss):
   def __init__(self, alpha=2, beta=5, name="heat_loss", **kwargs):
      super().__init__(name=name, **kwargs)
      self.alpha = alpha
      self.beta = beta


   def call(self, y_true, y_pred, sample_weight=None):
      weights = tf.where(y_true!=0, 1., 0.)
      weights = tf.exp(weights * self.alpha) # [1, e^alpha]
      return tf.reduce_mean(tf.square(y_true-y_pred) * weights) * self.beta
   

   def get_config(self):
      base_config = super().get_config()
      return {**base_config, "alpha": self.alpha, "beta": self.beta}


class PafLoss(tf.keras.losses.Loss):
   def __init__(self, alpha=2, name="paf_loss", **kwargs):
      super().__init__(name=name, **kwargs)
      self.alpha = alpha

   
   def call(self, y_true, y_pred, sample_weight=None):
      weights = tf.where(y_true!=0, 1., 0.)
      weights = tf.exp(weights * self.alpha)
      return tf.reduce_mean(tf.square(y_true-y_pred) * weights)
 

   def get_config(self):
      base_config = super().get_config()
      return {**base_config, "alpha": self.alpha}