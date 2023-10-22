import ijson, json
import numpy as np
from scipy.ndimage import maximum_filter
from data_loader import MetaData

class MapsParser:
   """Still in process"""
   def __init__(self, heatmaps, pafs, image=None):
      pass


   @staticmethod
   def parse_heatmaps(heatmaps, threshold=0.9):
      keypoints = []
      for i in range(heatmaps.shape[2]):
         data_max = maximum_filter(heatmaps[:,:,i], size=1)
         maxima = (heatmaps[:,:,i] == data_max)

         diff = ((data_max > threshold) & maxima)

         y, x = np.where(diff)

         kp = []
         kp.extend(zip(x, y))
         keypoints.append(kp)
      return keypoints

   @staticmethod
   def parse_pafs(keypoints, pafs, threshold=0.6):
      def check_connection(pafs, kp1, kp2, pair_id, threshold):
         x_start, y_start = kp1
         x_end, y_end = kp2
         dx = x_end - x_start
         dy = y_end - y_start

         norm = np.sqrt(dx**2 + dy**2)
         vec_x = dx / norm
         vec_y = dy / norm

         sample_x = np.linspace(kp1[0], kp2[0], num=int(norm))
         sample_y = np.linspace(kp1[1], kp2[1], num=int(norm))
         sample_x = sample_x.astype(int)
         sample_y = sample_y.astype(int)

         paf_x = pafs[sample_y, sample_x, pair_id*2]
         paf_y = pafs[sample_y, sample_x, pair_id*2+1]

         dot_product = vec_x*paf_x + vec_y*paf_y

         if np.mean(dot_product) > threshold:
            return True
         return False
     
      pairs = MetaData.pairs
      for idx, pair in enumerate(pairs):
         for kp1 in keypoints[pair[0]]:
            for kp2 in keypoints[pair[1]]: 
               if check_connection(pafs, kp1, kp2, idx, threshold):
                  print("connection!", kp1, kp2)



def extract_coco(filename, save_filename):
   """Extracts the most important information from COCO annotations and saves it to a JSON file. 
      >>> {"image_name.jpg": {"keypoints": list, "width": int, "height": int}}
   """
   annotations = {}
   with open(filename, 'r') as file:
      objects = ijson.items(file, 'annotations.item')
      for obj in objects:
         is_crowd = obj["iscrowd"]
         if is_crowd:
            continue
         image_id = obj["image_id"]
         image_id = "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
         keypoints = obj["keypoints"]
         keypoints = [float(x) for x in keypoints]
         if image_id not in annotations:
            annotations[image_id] = {"keypoints": keypoints}
         else:
            annotations[image_id]["keypoints"].extend(keypoints)

   with open(filename, 'r') as file:
      objects = ijson.items(file, 'images.item')
      for obj in objects:
         image_id = obj["id"]
         image_id = "0"*(12-len(str(image_id))) + str(image_id) + ".jpg"
         if image_id in annotations:
            annotations[image_id]["height"] = obj["height"]
            annotations[image_id]["width"] = obj["width"]

   with open(f"{save_filename}", 'w') as f:
      json.dump(annotations, f)



