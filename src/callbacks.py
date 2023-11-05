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

      # take first batch, then last layer
      pafs_pred = pafs_pred[0]
      heatmaps_pred = heatmaps_pred[0]
      MapsCompareCallback.compare_maps(images[0], 
                                         heatmaps_true[0], 
                                         heatmaps_pred,
                                         filename=f"{self.folder}{epoch}_heatmaps")
      MapsCompareCallback.compare_maps(images[0], 
                                         pafs_true[0], 
                                         pafs_pred,
                                         filename=f"{self.folder}{epoch}_pafs")


   @staticmethod
   def compare_maps(image, maps_true, maps_pred, filename, aplha=0.6):
      if type(maps_pred) != np.ndarray:
         maps_pred = maps_pred.numpy()
      ch = maps_true.shape[-1]
      rows = int(ch**0.5)
      cols = ch//rows

      traget_size = (image.shape[1], image.shape[0])
      # denormalize
      image = (image+1)*127.5
      if np.min(maps_true) < -0.9:
         maps_true = (maps_true+1)*127.5
         maps_pred = (maps_pred+1)*127.5
      elif np.max(maps_true) < 1.1:
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



class Checkpoint(tf.keras.callbacks.Callback):
   def __init__(self, ckpt_manager, monitor="val_loss", save_best_only=False, verbose=1):
      super().__init__()
      self.manager = ckpt_manager
      
      self.monitor = monitor
      self.monitor_value = float("inf")
      self.save_best_only = save_best_only

      self.verbose = verbose


   def on_epoch_end(self, _, logs=None):
      current_val_loss = logs.get(self.monitor)

      if self.save_best_only:
         if current_val_loss < self.monitor_value:
            if self.verbose:
               print(f"{self.monitor} improved from {self.monitor_value} to {current_val_loss}. Saving...")

            self.manager.save()  
            self.monitor_value = current_val_loss
         elif self.verbose:
            print(f"{self.monitor} didn't improved from {self.monitor_value}. Saving...")
      else:
         print("Saving model...")
         self.manager.save()  



class TensorBoard(tf.keras.callbacks.Callback):
   def __init__(self, log_dir, **kwargs):
      super().__init__(**kwargs)
      self.log_dir = log_dir
      self.file_writer = tf.summary.create_file_writer(self.log_dir)


   def on_train_batch_end(self, _, logs=None):
      logs = logs or {}
      for name, value in logs.items():
         if name in ["batch", "size", "evaluation_loss_vs_iterations"]:
            continue
         name = f"Steps/{name}"
         with self.file_writer.as_default():
            tf.summary.scalar(name, value, step=self.model.optimizer.iterations)
      self.file_writer.flush()


   def on_epoch_end(self, epoch, logs=None):
      logs = logs or {}
      for name, value in logs.items():
         if name in ["batch", "size", "evaluation_loss_vs_iterations"]:
            continue
         name = f"Epochs/{name}"
         with self.file_writer.as_default():
            tf.summary.scalar(name, value, step=epoch)
      self.file_writer.flush()



class ProgressBar(tf.keras.callbacks.Callback):
   def __init__(self, metrics):
      super().__init__()
      self.metrics = metrics
      self.progbar = None

   def on_epoch_begin(self, epoch, logs=None):
      print(f"Epoch {epoch+1}/{self.epochs}")
      self.averages = {metric: 0 for metric in self.metrics if not metric.startswith("val")}
      self.current_epoch = epoch
      self.progbar = tf.keras.utils.Progbar(target=self.steps, stateful_metrics=self.metrics)


   def on_train_begin(self, logs=None):
      self.epochs = self.params["epochs"]
      self.steps = self.params['steps']


   def on_train_batch_end(self, batch, logs=None):
      alpha = 1/(batch+1) if batch !=0 else 1

      values = [(k, v) for k, v in logs.items() if k in self.metrics]
      # calculate moving averange
      for k, v in values:
         self.averages[k] = (1-alpha)*self.averages[k] +  alpha*v

      averages = [(k, v) for k, v in self.averages.items()]
      self.progbar.update(batch, values=averages)


   def on_epoch_end(self, _, logs=None):
      metrics = [(k, v) for k, v in self.averages.items()]

      self.progbar.update(self.steps, values=metrics, finalize=False)


   def on_test_begin(self, logs=None):
      val_metrics = {metric: 0 for metric in self.metrics if metric.startswith("val")}
      self.averages.update(val_metrics)


   def on_test_batch_end(self, batch, logs=None):
      alpha = 1/(batch+1) if batch !=0 else 1

      values = [(k, v) for k, v in logs.items() if k in self.metrics]

      for k, v in values:
         self.averages[k] = (1-alpha)*self.averages[k] + alpha*v


   def on_test_end(self, logs=None):
      metrics = [(k, v) for k, v in self.averages.items()]

      self.progbar.update(self.steps, values=metrics, finalize=True)