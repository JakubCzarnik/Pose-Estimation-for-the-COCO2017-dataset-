import tensorflow as tf
import numpy as np
import cv2

class MapsCompareCallback(tf.keras.callbacks.Callback):
   def __init__(self, model, generator, folder):
      self.model = model
      self.generator = generator
      self.folder = folder


   def on_epoch_end(self, epoch, logs=None):
      idx = np.random.randint(0, len(self.generator), size=1)[0]
      images, (heatmaps_true, pafs_true) = self.generator[idx]
      heatmaps_pred, pafs_pred = self.model(images)
      MapsCompareCallback.compare_maps(images[0], 
                                         heatmaps_true[0], 
                                         heatmaps_pred[0],
                                         filename=f"{self.folder}{epoch}_heatmaps")
      MapsCompareCallback.compare_maps(images[0], 
                                         pafs_true[0], 
                                         pafs_pred[0],
                                         filename=f"{self.folder}{epoch}_pafs")


   @staticmethod
   def compare_maps(image, maps_true, maps_pred, filename, aplha=0.6):
      if type(maps_pred) != np.ndarray:
         maps_pred = maps_pred.numpy()
      c = maps_true.shape[-1]
      rows = int(c**0.5)
      cols = c//rows

      traget_size = (image.shape[1], image.shape[0])
      # denormalize
      if np.max(image) < 1.1:
         image = image*255
      if np.min(maps_true) < -0.9:
         maps_true = (maps_true+1)*127.5
         maps_pred = (maps_pred+1)*127.5
      else:
         maps_true = maps_true*255
         maps_pred = maps_pred*255
      # save outputs
      for maps, title in [(maps_true, "true"), (maps_pred, "pred")]:
         grid = np.zeros((traget_size[1]*rows, traget_size[0]*cols, 3), dtype=np.uint8)
         for j in range(rows):
            for k in range(cols):
               map = cv2.resize(maps[..., j*(rows)+k], traget_size)
               map = cv2.applyColorMap(np.uint8(map), cv2.COLORMAP_JET)
               combined = aplha*map + (1-aplha)*image
               h_start, h_end = traget_size[1]*j, traget_size[1]*(j+1)
               w_start, w_end = traget_size[0]*k, traget_size[0]*(k+1)
               grid[h_start:h_end, w_start:w_end, :] = combined
         grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
         cv2.imwrite(f"{filename}_{title}.jpg", grid)








