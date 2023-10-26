import tensorflow as tf

class HeatMSE(tf.keras.losses.Loss):
   """An extended version of the MSE loss function with weights. 
      Weights are calculated by using the exp() function on the 
      values of the y_true brought to the range ((0-beta)*alpha, (1-beta)*alpha),
      and then multiplied by gamma for balancing among other losses.

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
   def __init__(self, alpha=1, beta=0.05, gamma=1, clip_weights=300., name="heat_mse", *args, **kwargs):
      super().__init__(name=name, *args, **kwargs)
      self.alpha = alpha
      self.beta = beta
      self.gamma = gamma # gamma is used for balance between losses
      self.clip_weights = clip_weights


   def get_config(self):
      config = super().get_config()
      config.update({"alpha": self.alpha,
                     "beta": self.beta,
                     "gamma": self.gamma,
                     "clip_weights": self.clip_weights})
      return config


   def call(self, y_true, y_pred):
      weights = (y_true - self.beta) * self.alpha
      weights = tf.exp(weights) * self.gamma
      weights = tf.clip_by_value(weights, 0., self.clip_weights)
      return tf.reduce_mean(tf.square(y_pred - y_true) * weights)
   

class PAFMSE(HeatMSE):
   """An extended version of the MSE loss function with weights. 
      Weights are calculated by using the exp() function on the 
      binary mask of y_true (1 where there is a vector, 0 elsewhere), 
      which is then brought to the range ((0-beta)*alpha, (1-beta)*alpha),
      and then multiplied by gamma for balancing among other losses.

      You can adjust the ratio of weights by selecting the alpha and beta parameter. 
      In this way, the model focuses more on key areas.

      The smaller the beta, the smallest weights converge to 1 and the largest to exp(alpha). 
      The larger the beta, the smallest weights converge to zero and the largest to 1
      >>> example on alpha=4, beta=0.01:
      >>> heatmap[..., 0] -> [[0.   0.   0.   0.   0.  ]
                              [0.   0.71 0.71 0.71 0.  ]
                              [0.   0.71 0.71 0.71 0.  ]
                              [0.   0.71 0.71 0.71 0.  ]
                              [0.   0.   0.   0.   0.  ]]
      >>> weights[..., 0] -> [[ 0.96  0.96  0.96  0.96  0.96]
                              [ 0.96 52.46 52.46 52.46  0.96]
                              [ 0.96 52.46 52.46 52.46  0.96]
                              [ 0.96 52.46 52.46 52.46  0.96]
                              [ 0.96  0.96  0.96  0.96  0.96]]
   """
   def __init__(self, alpha=1, beta=0.05, gamma=1, clip_weights=300, name="paf_mse", *args, **kwargs):
      super().__init__(alpha, beta, gamma, clip_weights, name, *args, **kwargs)

   
   def call(self, y_true, y_pred):
      weights = tf.where(y_true!=0, 1., 0.)
      weights = (weights - self.beta) * self.alpha
      weights = tf.exp(weights) * self.gamma
      weights = tf.clip_by_value(weights, 0., self.clip_weights)
      return tf.reduce_mean(tf.square(y_pred - y_true) * weights)