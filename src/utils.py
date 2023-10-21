import ijson, json
import numpy as np
from scipy.ndimage import maximum_filter

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


def load_annotations(annotations_path="annotations.json"):
   """Returns train/test annotations from annotations file.
   """
   with open(annotations_path) as file:
      annotations = json.load(file)
   return annotations




### rm
def calculate_points(heatmap, threshold=0.5):
   keypoints = []
   for i in range(heatmap.shape[2]):
        data_max = maximum_filter(heatmap[:,:,i], size=5)
        maxima = (heatmap[:,:,i] == data_max)

        diff = ((data_max > threshold) & maxima)

        y, x = np.where(diff)

        kp = []
        kp.extend(zip(x, y))
        keypoints.append(kp)
   return keypoints


def check_connection(pafs, kp1, kp2, idx, threshold=0.6):
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

   paf_x = pafs[sample_y, sample_x, idx*2]
   paf_y = pafs[sample_y, sample_x, idx*2+1]

   dot_product = vec_x*paf_x + vec_y*paf_y

   if np.mean(dot_product) > threshold:
      return True
   return False


def decode(image, heatmap, pafs):
   import matplotlib.pyplot as plt

   pairs = [(0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
            (4, 6), (3, 5), (6, 12), (5, 11)]


   keypoints = calculate_points(heatmap)
   plt.figure()
   plt.imshow(image)

   for idx, pair in enumerate(pairs):
      for kp1 in keypoints[pair[0]]:
         for kp2 in keypoints[pair[1]]: 
            if check_connection(pafs, kp1, kp2, idx):
               plt.plot([kp1[0], kp2[0]], [kp1[1], kp2[1]], 'r')


   for joint in keypoints:
      for point in joint:
         plt.plot(point[0], point[1], 'bo')

   plt.show()