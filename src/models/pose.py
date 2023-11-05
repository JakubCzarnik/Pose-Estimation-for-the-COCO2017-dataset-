import tensorflow as tf
from tensorflow.keras import Model


class PoseDetector(Model):
   def __init__(self, model, config, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.model = model
      self.stage_weights = config.stage_weights


   def call(self, inputs, training=False):
      heatmaps, pafs = self.model(inputs, training=training)
      return heatmaps[-1], pafs[-1]


   def compile(self, optimizer, heat_loss, paf_loss, *args, **kwargs):
      super().compile(*args, **kwargs)
      self.optimizer = optimizer
      self.heat_loss = heat_loss
      self.paf_loss = paf_loss


   def train_step(self, data):
      images, (heatmap_true, paf_true) = data

      with tf.GradientTape() as tape:
         heatmaps_pred, pafs_pred = self.model(images, training=True)

         sum_loss = 0
         loss_dict = {}
         for i, hmap in enumerate(heatmaps_pred):
            loss = self.heat_loss(heatmap_true, hmap) * self.stage_weights[i]
            loss_dict[f"heat_{i}"] = loss
            sum_loss += loss

         for i, paf in enumerate(pafs_pred):
            loss = self.paf_loss(paf_true, paf) * self.stage_weights[i]
            loss_dict[f"paf_{i}"] = loss
            sum_loss += loss

         loss_dict[f"loss"] = sum_loss
         
      grads = tape.gradient(sum_loss, self.model.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
      return loss_dict
   

   def test_step(self, data):
      images, (heatmap_true, paf_true) = data
      heatmaps_pred, pafs_pred = self.model(images, training=False)

      sum_loss = 0
      loss_dict = {}
      for i, hmap in enumerate(heatmaps_pred):
         loss = self.heat_loss(heatmap_true, hmap) * self.stage_weights[i]
         loss_dict[f"heat_{i}"] = loss
         sum_loss += loss

      for i, paf in enumerate(pafs_pred):
         loss = self.paf_loss(paf_true, paf) * self.stage_weights[i]
         loss_dict[f"paf_{i}"] = loss
         sum_loss += loss

      loss_dict[f"loss"] = sum_loss
      return loss_dict