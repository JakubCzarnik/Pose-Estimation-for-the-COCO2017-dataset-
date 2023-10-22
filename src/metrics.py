import tensorflow as tf

class ExponentialMSE(tf.keras.losses.Loss):
   """An extended version of the MSE loss function with weights. 
      Weights are calculated by using the exp() function on the 
      values of the y_true brought to the range ((0-beta)*alpha, (1-beta)*alpha). 
      You can adjust the ratio of weights by selecting the alpha and beta parameter. 
      In this way, the model focuses more on key areas.

      The smaller the beta, the smallest weights converge to 1 and the largest to exp(alpha). 
      The larger the beta, the smallest weights converge to zero and the largest to 1
      >>> example on alpha=4, beta=0.01:
      >>> heatmap[..., 0] -> [[0.   0.02 0.04 0.02 0.  ]
                              [0.02 0.21 0.46 0.21 0.02]
                              [0.04 0.46 1.   0.46 0.04]
                              [0.02 0.21 0.46 0.21 0.02]
                              [0.   0.02 0.04 0.02 0.  ]]
      >>> weights[..., 0] -> [[ 0.96  1.04  1.15  1.04  0.96]
                              [ 1.04  2.22  6.    2.22  1.04]
                              [ 1.15  6.   52.46  6.    1.15]
                              [ 1.04  2.22  6.    2.22  1.04]
                              [ 0.96  1.04  1.15  1.04  0.96]]
   """
   def __init__(self, alpha=1, beta=0.1, name="exponential_mse", *args, **kwargs):
      super().__init__(name=name, *args, **kwargs)
      self.alpha = alpha
      self.beta = beta


   def get_config(self):
      config = super().get_config()
      config.update({"alpha": self.alpha,
                     "beta": self.beta})
      return config


   def call(self, y_true, y_pred):
      # note that heatmaps are in range (0, 1) and PAFs in (-1, 1)
      # bring to range x=((0-beta)*alpha, (1-beta)*alpha)
      weights = (tf.abs(y_true) - self.beta) * self.alpha
      # and then put into e^x
      weights = tf.exp(weights)
      return tf.reduce_mean(tf.square(y_pred - y_true) * weights)