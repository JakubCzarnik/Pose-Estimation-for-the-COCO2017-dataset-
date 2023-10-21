import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from utils import load_annotations
from callbacks import *


class MetaData:
   pairs = np.array([(0, 1), (0, 2), (1, 3), (2, 4), # head
         (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # body
         (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), # legs
         (0,17), (11, 17), (12, 17)], dtype=np.int8) # head-body-legs connections
   n_keypoints = 18 # 17 + 1 (additional)

   def __init__(self, 
                image, 
                annotation, 
                sigma,
                vec_width,
                target_image_size=(386, 386), 
                maps_size=(48, 48)):
      self.original_width = image.shape[1]
      self.original_height = image.shape[0] 
      self.annotation = MetaData.preprocess_annotations(annotation, image.shape[:2], maps_size)
      self.image = cv2.resize(image, target_image_size)/255
      self.data = MetaData.create_maps(self.annotation, maps_size, sigma, vec_width)


   def return_data(self):
      return self.image, self.data


   @staticmethod
   def preprocess_annotations(annotation, img_size, target_size):
      annotation = np.array(annotation["keypoints"], dtype=np.int16)
      annotation = annotation.reshape(-1, 17, 3)

      new_annotation = np.zeros((annotation.shape[0], MetaData.n_keypoints, 3), 
                                dtype=np.int16)
      for i, _ in enumerate(annotation):
         new_annotation[i, :17, :3] = annotation[i]
         left_arm = new_annotation[i, 5, :]
         right_arm = new_annotation[i, 6, :]
         # append chest point     
         new_annotation[i, 17, :] = (left_arm+right_arm)/2
         new_annotation[i, 17, 2] = 2 if new_annotation[i, 17, 2] > 1 else 0

      mask = new_annotation[..., 2] == 0
      new_annotation[mask, :2] = [0,0]
      # resize
      new_annotation[:, :, 0] = new_annotation[:, :, 0] * target_size[1] / img_size[1]
      new_annotation[:, :, 1] = new_annotation[:, :, 1] * target_size[0] / img_size[0]
      return new_annotation


   @staticmethod
   def create_maps(annotations, maps_size, sigma, vec_width):
      pairs =  MetaData.pairs
      heatmaps = np.zeros((*maps_size, MetaData.n_keypoints), dtype=np.float32)
      pafs = np.zeros((*maps_size, 2*len(pairs)), dtype=np.float32)
      
      for keypoints in annotations:
         # heatmaps
         for i, keypoint in enumerate(keypoints):
            if keypoint[2] > 0:
               heatmaps[:, :, i] = MetaData.create_heatmap(heatmaps[:, :, i],
                                                           keypoint[:2],
                                                           sigma)
         # Part Affiniti Fields
         for i in range(len(pairs)):
            pair = pairs[i]
            kps = [keypoints[pair[0]], keypoints[pair[1]]]
            if kps[0][2] > 0 and kps[1][2] > 0:
               pafs[:, :, 2*i:2*i+2] = MetaData.create_paf(pafs[:, :, 2*i:2*i+2],
                                                           kps,
                                                           vec_width)
      # Clipping
      heatmaps = np.clip(heatmaps, 0, 1)
      pafs = np.clip(pafs, -1, 1)
      return heatmaps, pafs


   @staticmethod
   def create_heatmap(heatmap, keypoint, sigma):
      size = heatmap.shape
      map = np.zeros(size, dtype=np.float32)
      
      x = np.arange(0, size[1], 1, dtype=np.int16)
      y = np.arange(0, size[0], 1, dtype=np.int16)
      xx, yy = np.meshgrid(x, y)
      
      x_kp, y_kp = keypoint
      d = (xx - x_kp)**2 + (yy - y_kp)**2
      exponent = d / 2.0 / sigma / sigma
      mask = exponent <= 4.6052  # threshold, ln(100)
      map[mask] += np.exp(-exponent[mask])
      map[map > 1.0] = 1.0
      
      heatmap += map
      return heatmap


   @staticmethod
   def create_paf(pafs, keypoints, width):
      size = pafs.shape[:2]
      map = np.zeros((*size, 2), dtype=np.float32)

      x_start, y_start = keypoints[0][:2]
      x_end, y_end = keypoints[1][:2]

      dx = x_end - x_start
      dy = y_end - y_start
      norm = np.sqrt(dx**2 + dy**2)
      dx /= (norm + 1e-5)  
      dy /= (norm + 1e-5) 

      sample_x = np.linspace(x_start, x_end, num=int(norm), dtype=np.int16)
      sample_y = np.linspace(y_start, y_end, num=int(norm), dtype=np.int16)
      
      # create a grid of offsets
      w_range = np.arange(-width//2, width//2+1, dtype=np.int16)
      h_range = np.arange(-width//2, width//2+1, dtype=np.int16)
      offset_grid = np.array(np.meshgrid(w_range, h_range), dtype=np.int16).T.reshape(-1, 2)

      # iterate over the line between the keypoints
      for j in range(int(norm)):
         paf_x = sample_x[j]
         paf_y = sample_y[j]
         if 0 <= paf_x < size[1] and 0 <= paf_y < size[0]:
               coords = np.array([paf_x, paf_y], dtype=np.int16) + offset_grid
               valid_coords_mask = np.all((coords >= 0) & (coords < size), axis=1)
               valid_coords = coords[valid_coords_mask]
               map[valid_coords[:, 1], valid_coords[:, 0], :] = [dx, dy]

      pafs += map
      return pafs



class DataGenerator(tf.keras.utils.Sequence):
   def __init__(self, 
                annotations_path:str, 
                data_folder:str, 
                batch_size:int, 
                batches:int, 
                shuffle:bool=True,
                sigma:float=0.8,
                vec_width:int=3,
                image_size=(386,386),
                maps_size=(48,48)):
      self.annotations = load_annotations(annotations_path)
      self.data_folder = data_folder
      self.batch_size = batch_size
      self.batches = batches
      self.shuffle = shuffle
      self.sigma = sigma
      self.vec_width = vec_width
      self.image_size = image_size
      self.maps_size = maps_size

      self.on_epoch_end(create_indices=True)


   def __len__(self):
      return len(self.indices) // self.batch_size


   def on_epoch_end(self, create_indices=False):
      """Choices new indices from annotations.
         Notice: this method selecting {num_batches} batches randomly from whole dataset."""
      if self.shuffle or create_indices:
         data_size = len(self.annotations)
         samples = int(self.batch_size*self.batches)
         self.indices = np.random.choice(range(data_size), size=samples, replace=False)


   def __getitem__(self, index):
      start_idx = index * self.batch_size
      end_idx = (index + 1) * self.batch_size
      batch_indices = self.indices[start_idx:end_idx]

      X = np.zeros((self.batch_size, *self.image_size, 3), 
                   dtype=np.float32)
      y_heatmap = np.zeros((self.batch_size, *self.maps_size, MetaData.n_keypoints), 
                           dtype=np.float32) 
      y_paf = np.zeros((self.batch_size, *self.maps_size, 2*len(MetaData.pairs)), 
                       dtype=np.float32)

      for i, idx in enumerate(batch_indices):
         image_name = list(self.annotations.keys())[int(idx)]
         image = cv2.imread(f"{self.data_folder}{image_name}")
         
         annotation = self.annotations[image_name]
         metadata = MetaData(image, annotation, self.sigma, self.vec_width)

         image, (heatmap, pafs) = metadata.return_data()
         X[i] = image
         y_heatmap[i] = heatmap
         y_paf[i] = pafs

      return X, [y_heatmap, y_paf]



if __name__ == "__main__":
   np.random.seed(676)
   train_gen = DataGenerator("train_annotations.json", 
                          "D:/COCO 2017/train2017/", 
                          batch_size=16, 
                          batches=350)

   import time
   t1 = time.time()
   mean = []
   for i, batch in enumerate(train_gen):
      if True: # visualize
         image, (heat, pafs) = batch
         MapsCompareCallback.compare_maps(image[0], 
                                         heat[0], 
                                         heat[0],
                                         "heat")
         MapsCompareCallback.compare_maps(image[0], 
                                         pafs[0], 
                                         pafs[0],
                                         "pafs")
         break
      else: # timeit
         image, (heat, pafs) = batch
         t2 = time.time()
         mean.append(t2-t1)
         print(f"{t2-t1=:.2f}  {np.mean(mean)=:.2f} ")
         print(f"{(image.nbytes+heat.nbytes+pafs.nbytes)/1048576:.2f}, mb ")
         t1=t2
         if i == 20:
            break

  