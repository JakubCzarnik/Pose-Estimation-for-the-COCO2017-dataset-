import cv2, json
import numpy as np
import tensorflow as tf
from callbacks import *
import albumentations as A



class MetaData:
   target_image_size = (256, 256)
   target_maps_size = (64, 64)
   sigma = 0.9
   vec_width = 2
   n_keypoints = 18 # 17 + 1 (chest-additional)
   pairs = np.array([(17,0), # head-body     
         (0, 1), (0, 2), (1, 3), (2, 4), # head
         (17, 5),  (17, 6), (5, 7), (7, 9), (6, 8), (8, 10), # body,arms
         (11, 17), (17, 12), # body-legs
         (12, 11), (11, 13), (12, 14), (13, 15), (14, 16), # legs
         ], dtype=np.uint8) # The order is important for connecting 

   transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.75),
    A.SmallestMaxSize(max_size=target_image_size[0]),
    A.RandomCrop(width=target_image_size[1], height=target_image_size[0], p=0.5),
    A.GaussNoise(var_limit=(10., 50.), p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.CLAHE(clip_limit=1),
    A.ImageCompression(quality_lower=75),
    A.Resize(width=target_image_size[1], height=target_image_size[0]),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
   ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


   def __init__(self, image, annotation):
      self.original_width = image.shape[1]
      self.original_height = image.shape[0]
      self.image, self.annotation = MetaData.preprocess_data(image, annotation)
      self.maps = MetaData.create_maps(self.annotation)


   def return_data(self):
      return self.image, self.maps


   @staticmethod
   def preprocess_data(image, annotation):
      annotation = np.array(annotation["keypoints"], dtype=np.int32)
      annotation = annotation.reshape(-1, 17, 3)

      keypoints = np.zeros((annotation.shape[0], MetaData.n_keypoints, 3), 
                                dtype=np.int32)
      keypoints[:, :17, :3] = annotation

      # calculate chest point
      left_arm = keypoints[:, 5, :3] # x,y,v
      right_arm = keypoints[:, 6, :3] # x,y,v
      keypoints[:, 17, :3] = (left_arm+right_arm)/2
      keypoints[:, 17, 2] = np.where(keypoints[:, 17, 2]>1, 2, 0)
  
      # check validity and put (0, 0) on v=0
      mask = keypoints[..., 2] == 0
      keypoints[mask, :2] = [0,0]

      # transform
      keypoints = keypoints.reshape(-1, 3)
      transformed = MetaData.transform(image=image, keypoints=keypoints)
      keypoints = transformed['keypoints']
      image = transformed['image']
      keypoints = np.array(keypoints, dtype=np.float64).reshape(-1, MetaData.n_keypoints, 3)
      
      # rescale keypoints to map target size
      keypoints[..., 0] *= (MetaData.target_maps_size[1]/MetaData.target_image_size[1])
      keypoints[..., 1] *= (MetaData.target_maps_size[0]/MetaData.target_image_size[0])
 
      keypoints = keypoints.astype(np.int32)
      return image, keypoints


   @staticmethod
   def create_maps(annotations):
      pairs =  MetaData.pairs
      maps_size = MetaData.target_maps_size
      heatmaps = np.zeros((*maps_size, MetaData.n_keypoints), dtype=np.float32)
      pafs = np.zeros((*maps_size, 2*len(pairs)), dtype=np.float32)
      
      for keypoints in annotations:
         # heatmaps
         for i, keypoint in enumerate(keypoints):
            if keypoint[2] > 0:
               heatmaps[:, :, i] = MetaData.create_heatmap(heatmaps[:, :, i],
                                                           keypoint[:2])
         # Part Affiniti Fields
         for i in range(len(pairs)):
            pair = pairs[i]
            kps = [keypoints[pair[0]], keypoints[pair[1]]]
            if kps[0][2] > 0 and kps[1][2] > 0:
               pafs[:, :, 2*i:2*i+2] = MetaData.create_paf(pafs[:, :, 2*i:2*i+2],
                                                           kps)
      # Clipping
      heatmaps = np.clip(heatmaps, 0, 1)
      pafs = np.clip(pafs, -1, 1)
      return heatmaps, pafs


   @staticmethod
   def create_heatmap(heatmap, keypoint):
      size = heatmap.shape
      map = np.zeros(size, dtype=np.float32)
      
      x = np.arange(0, size[1], 1, dtype=np.int16)
      y = np.arange(0, size[0], 1, dtype=np.int16)
      xx, yy = np.meshgrid(x, y)
      
      x_kp, y_kp = keypoint
      d = (xx - x_kp)**2 + (yy - y_kp)**2
      exponent = d / 2.0 / MetaData.sigma / MetaData.sigma
      threshold = 4.6052  # ln(100)
      mask = exponent <= threshold  
      exponent = np.clip(exponent, 0, threshold)
      map[mask] += np.exp(-exponent[mask])
      
      heatmap += map
      return heatmap


   @staticmethod
   def create_paf(pafs, keypoints):
      size = pafs.shape[:2]
      width = MetaData.vec_width
      map = np.zeros((*size, 2))
      x1, y1, _ = keypoints[0]
      x2, y2, _ = keypoints[1]
      dx = x2 - x1
      dy = y2 - y1

      norm = np.sqrt(dx**2 + dy**2)+1e-5
      vec_x = dx / norm
      vec_y = dy / norm

      if x1 != x2:
         # y = ax + b
         a = (y1-y2)/(x1-x2) 
         b = y1 - a*x1
         x_cords = np.linspace(x1, x2, num=90, dtype=np.float32)
         y_cords = x_cords*a + b
         for x, y in zip(x_cords, y_cords):
            w_range = np.arange(-width//2, width//2+1)
            h_range = np.arange(-width//2, width//2+1)
            w_coords = np.clip(x + w_range, 0, map.shape[1]-1).astype(np.int16)
            h_coords = np.clip(y + h_range, 0, map.shape[0]-1).astype(np.int16)
            map[h_coords[:, None], w_coords] = vec_x, vec_y

      else:
         y_cords = np.linspace(y1, y2, num=90, dtype=np.float32)
         y_cords = np.clip(y_cords, 0, map.shape[0]-1)
         for y in y_cords:
            w_range = np.arange(-width//2, width//2+1)
            w_coords = np.clip(x1 + w_range, 0, map.shape[1]-1).astype(np.int16)
            map[np.int16(y), w_coords, 1] = vec_y
         
      pafs += map
      return pafs


   
class DataGenerator(tf.keras.utils.Sequence):
   batch_size = 16
   def __init__(self, 
                annotations_path:str, 
                data_folder:str, 
                batches:int, 
                shuffle:bool=True):
      self.annotations = DataGenerator.load_annotations(annotations_path)
      self.data_folder = data_folder
      self.batches = batches
      self.shuffle = shuffle

      self.on_epoch_end(create_indices=True)


   def __len__(self):
      return len(self.indices) // DataGenerator.batch_size


   @staticmethod
   def load_annotations(annotations_path="annotations.json"):
      """Returns train/test annotations from annotations file.
      """
      with open(annotations_path) as file:
         annotations = json.load(file)
      return annotations


   def on_epoch_end(self, create_indices=False):
      """Choices new indices from annotations.
         Notice: this method selecting {num_batches} batches randomly from whole dataset.
      """
      if self.shuffle or create_indices:
         data_size = len(self.annotations)
         samples = int(DataGenerator.batch_size*self.batches)
         self.indices = np.random.choice(range(data_size), size=samples, replace=False)


   def __getitem__(self, index):
      start_idx = index * DataGenerator.batch_size
      end_idx = (index + 1) * DataGenerator.batch_size
      batch_indices = self.indices[start_idx:end_idx]

      
      X = np.zeros((DataGenerator.batch_size, *MetaData.target_image_size, 3), 
                   dtype=np.float32)
      y_heatmap = np.zeros((DataGenerator.batch_size, *MetaData.target_maps_size, MetaData.n_keypoints), 
                           dtype=np.float32) 
      y_paf = np.zeros((DataGenerator.batch_size, *MetaData.target_maps_size, 2*len(MetaData.pairs)), 
                       dtype=np.float32)

      for i, idx in enumerate(batch_indices):
         image_name = list(self.annotations.keys())[int(idx)]
         image = cv2.imread(f"{self.data_folder}{image_name}")
         
         annotation = self.annotations[image_name]
         metadata = MetaData(image, annotation)

         image, (heatmap, pafs) = metadata.return_data()
         X[i] = image
         y_heatmap[i] = heatmap
         y_paf[i] = pafs

      return X, [y_heatmap, y_paf]



if __name__ == "__main__":
   np.random.seed(676)
   #np.random.seed(6777)
   train_gen = DataGenerator("train_annotations.json", 
                             "D:/COCO 2017/train2017/", 
                              batches=350)

   import time
   import matplotlib.pyplot as plt
   t1 = time.time()
   mean = []
   for i, batch in enumerate(train_gen):
      if True: # visualize
         image, (heat, pafs) = batch
         image, heat, pafs = image[0], heat[0], pafs[0]
         plt.imshow((image*255).astype(np.uint8))
         plt.show()
         MapsCompareCallback.compare_maps(image, 
                                         heat, 
                                         heat,
                                         "heat")
         MapsCompareCallback.compare_maps(image, 
                                         pafs, 
                                         pafs,
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

  