import tensorflow as tf

class ExponentialMSE(tf.keras.losses.Loss):
   """An extended version of the MSE loss function with weights. 
   Weights are calculated by using the exp() function on the 
   absolute value of the y_true brought to the range (0-beta, 1-beta). As a result, areas with higher values
   receive more weight. You can adjust the ratio of weights by 
   selecting the alpha parameter. In this way, the model focuses more on key 
   areas.
   The smaller the beta, the smallest weights converge to 1 and the largest to 
   exp(alpha). The larger the beta, the smallest weights converge to zero and the largest to 1
   >>> example on alpha=1, beta=0.5:
   >>> heatmap[..., 0] -> [[0.   0.02 0.04 0.02 0.  ]
                           [0.02 0.21 0.46 0.21 0.02]
                           [0.04 0.46 1.   0.46 0.04]
                           [0.02 0.21 0.46 0.21 0.02]
                           [0.   0.02 0.04 0.02 0.  ]]
                            
   >>> weights[..., 0] -> [[0.61 0.62 0.63 0.62 0.61]
                           [0.62 0.75 0.96 0.75 0.62]
                           [0.63 0.96 1.65 0.96 0.63]
                           [0.62 0.75 0.96 0.75 0.62]
                           [0.61 0.62 0.63 0.62 0.61]]
   As we can see, the ratio of the lowest value to the highest value is equal to 0.61/1.61 ~ 0.37
   on alpha=2 the ratio is equal to 0.37/2.72 ~ 0.13
   on aplha=3 is already 0.22/4.48 ~ 0.05
   """
   def __init__(self, alpha:float=0, beta=0.1, name="weighted_mse"):
      super().__init__(name=name)
      self.alpha = alpha
      self.beta = beta


   def get_config(self):
      config = super().get_config()
      config.update({"alpha": self.alpha,
                     "beta": self.beta})
      return config


   def call(self, y_true, y_pred):
      # note that heatmaps are in range (0, 1) and PAFs in (-1, 1)
      # bring to range ((0-beta)*alpha, (1-beta)*alpha)
      weights = (tf.abs(y_true) - self.beta) * self.alpha
      # e^x
      weights = tf.exp(weights)
      return tf.reduce_mean(tf.square(y_pred - y_true) * weights)