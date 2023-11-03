import ijson, json, os
import numpy as np
import tensorflow as tf
from skimage.feature import peak_local_max
from data_loader import MetaData


class MapsParser:
   maxima_filter_size = 3
   heatmaps_threshold = 0.8
   pafs_threshold = 0.75
   conections_threshold = 4


   def __init__(self, heatmaps, pafs, image=None):
      self.heatmaps = heatmaps
      self.pafs = pafs
      self.image = image

      self.keypoints = MapsParser.parse_heatmaps(heatmaps)
      self.connections = MapsParser.parse_pafs(self.keypoints, self.pafs)

   @classmethod
   def update(cls, config):
      cls.maxima_filter_size = config.maxima_filter_size
      cls.heatmaps_threshold = config.heatmaps_threshold
      cls.pafs_threshold = config.pafs_threshold
      cls.conections_threshold = config.conections_threshold


   @staticmethod
   def parse_heatmaps(heatmaps):
      """This method extracts local maximums from the input heatmaps
         that exceed a certain threshold and returns them in the array of shape:
         (n_pairs, n_objects, (x,y,v)).
      """
      keypoints = [] 
      for i in range(heatmaps.shape[2]):
         kp = []

         local_maxima = peak_local_max(heatmaps[..., i], min_distance=MapsParser.maxima_filter_size)
         for coord in local_maxima:
            y, x = coord
            if heatmaps[..., i][y, x] > MapsParser.heatmaps_threshold:
               kp.append((x, y, 2))

         keypoints.append(kp)
      n_obj = max([len(list1) for list1 in keypoints])
      # (pair_id, n_pairs, (x,y,v))
      keypoints_arr = np.zeros((len(MetaData.pairs), n_obj, 3), np.int32)
      for i in range(len(keypoints)):
         if len(keypoints[i])>0:
            keypoints_arr[i, :len(keypoints[i])] = keypoints[i]
      return keypoints_arr


   @staticmethod
   def parse_pafs(keypoints, pafs):
      """This method connects keypoints where vectors in PAF's
         exceed a certain threshold and return them in array of shape:
         (n_pairs, n_objects, n_objects, 2, 3)

         Note: We have n_objects**2 connections because sometimes PAFs fail, 
         and in the worst case, all possible points will be connected with each other.
         
         For example:
         >>> If we have 5 nose keypoints and 5 chest keypoints, in the worst case we will receive
             5*5 = 25 connections.
      """
      def check_connection(pafs, kp1, kp2, pair_id):
         vec_x = kp2[0] - kp1[0]
         vec_y = kp2[1] - kp1[1]
         if vec_x == 0 and vec_y == 0:
            return True
         
         norm = np.sqrt(vec_x**2 + vec_y**2) 
         vec_x /= norm
         vec_y /= norm
         
         start_y = max(min(kp1[1], kp2[1]) - 2, 0)
         end_y = min(max(kp1[1], kp2[1]) + 3, pafs.shape[0])
         start_x = max(min(kp1[0], kp2[0]) - 2, 0)
         end_x = min(max(kp1[0], kp2[0]) + 3, pafs.shape[1])

         paf_values_x = pafs[start_y:end_y, start_x:end_x, 2*pair_id]
         paf_values_y = pafs[start_y:end_y, start_x:end_x, 2*pair_id+1]

         dot_product_x = paf_values_x * vec_x
         dot_product_y = paf_values_y * vec_y
         score = np.sum(dot_product_x + dot_product_y) / norm / (np.abs(vec_x) + np.abs(vec_y)+1e-4)
         score = score/(MetaData.vec_width+2)
         return score > MapsParser.pafs_threshold
     

      connections = np.zeros((len(MetaData.pairs), keypoints.shape[1], keypoints.shape[1], 2, 3), dtype=np.int32)
      for pair_id, pair in enumerate(MetaData.pairs):
         for id1, kp1 in enumerate(keypoints[pair[0]]):
            for id2, kp2 in enumerate(keypoints[pair[1]]): 
               if check_connection(pafs, kp1, kp2, pair_id):
                  connections[pair_id, id1, id2, ...] = (kp1, kp2)

      return connections
   

   @staticmethod
   def parse_connections(connections):
      """This method selects the best connections from all possible connections by minimizing
         the sum of distances. Additionally, it returns connections merged by objects in an array of shape:
         (n_objects, n_keypoints, 3)."""
      def get_best_sequence(list_points, actual_seq, depth=0):
         min_distance = float("inf")
         best_seq = None
         for connection in list_points[0]:
            path = np.copy(actual_seq)
            # if in acutal path second keypoint is connected then skip
            if np.isin(path[:, 1], connection[1]).all(axis=1).any():
               continue
            path[depth] = connection
            # if we are not at the end of sequence, we keep going
            distance = None
            if len(list_points) > 1:
               path, distance = get_best_sequence(list_points[1:], path, depth+1)

            # at the end we calculating sum of distances
            if distance is None:
               distance = np.sqrt(np.sum((path[:, 0, :2] - path[:, 1, :2]) ** 2, axis=1))
               distance = np.sum(distance)

            # update actual best sequence
            if distance < min_distance:
               best_seq = path
               min_distance = distance

         if depth > 0:
            return best_seq, min_distance
         # ending result
         if best_seq is None:
            return np.zeros_like(actual_seq, dtype=np.int32)
         return np.array(best_seq, dtype=np.int32)
      
      # 1. Select best fiting connections by minimum sum of distances 
      best_connections = np.zeros((*connections.shape[:2], 2, 3))

      for pair_id, pairs in enumerate(connections):
         extracted_connections = []
         for i in range(pairs.shape[0]):
            mask = np.all(pairs[i] != 0, axis=-1)
            cont = pairs[i][mask].reshape(-1, 2, 3)
            if cont.shape[0] != 0:
               extracted_connections.append(cont)

         starting_sequence = np.zeros((len(extracted_connections), 2, 3), dtype=np.int32)
         output = get_best_sequence(extracted_connections, starting_sequence)
         best_connections[pair_id, :output.shape[0]] = output
   
      # 2. Merge connections for each person
      merged_connections = np.zeros((connections.shape[1], len(MetaData.pairs), 3), dtype=np.int32)
      blocked_kp = np.zeros((len(MetaData.pairs)*connections.shape[1], 3))
      i = 0
      for obj_id in range(connections.shape[1]): 
         actual_instance = np.zeros((len(MetaData.pairs), 3), dtype=np.int32)
         for pair_id, (joint1_id, joint2_id) in enumerate(MetaData.pairs):
            selected_pair = False
            for con in best_connections[pair_id]:
               *_, v1 = con[0]
               *_, v2 = con[1]
               if v1*v2 == 0:
                  continue
               if np.any(np.isin(blocked_kp, con[1]).all(axis=(-1))):
                  continue
 
               if np.all(actual_instance==0):
                  actual_instance[joint1_id] = con[0]
                  actual_instance[joint2_id] = con[1]
                  blocked_kp[i, :] = con[1]
                  selected_pair = True
                  i += 1
               elif np.any(np.isin(actual_instance, con[0]).all(axis=(-1))) or \
                    np.any(np.isin(actual_instance, con[1]).all(axis=(-1))):
                  actual_instance[joint1_id] = con[0]
                  actual_instance[joint2_id] = con[1]
                  blocked_kp[i, :] = con[1]
                  selected_pair = True
                  i += 1
               if selected_pair:
                  break
         merged_connections[obj_id] = actual_instance

      # 3. Clear objects - select only those which have more than threshold
      zero_points = np.all(merged_connections == [0,0,0], axis=-1)
      zero_counts = np.sum(zero_points, axis=1)
      mask = MetaData.n_keypoints - zero_counts >=  MapsParser.conections_threshold
      cleared_connections = merged_connections[mask]
      return cleared_connections



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


def set_memory_growth():
   gpus = tf.config.list_physical_devices('GPU')
   if len(gpus) == 0:
      raise SystemError
   for gpu in gpus:
      print(gpu) 
      tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_annotations(config):
   if not os.path.isfile(config.extracted_train_annotations):
      extract_coco(config.train_annotations, save_filename=config.extracted_train_annotations)
   if not os.path.isfile(config.extracted_val_annotations):
      extract_coco(config.val_annotations, save_filename=config.extracted_val_annotations)



if __name__ == "__main__":
   from data_loader import DataGenerator
   import matplotlib.pyplot as plt

   #np.random.seed(676)
   np.random.seed(6777)
   #np.random.seed(321513)
   train_gen = DataGenerator("train_annotations.json", 
                             "D:/COCO 2017/train2017/", 
                              batches=1)
   for i, batch in enumerate(train_gen):
      image, (heat, pafs) = batch
      image, heat, pafs = image[0], heat[0], pafs[0]

      mp = MapsParser(heat, pafs, image)

      conn  = mp.parse_connections(mp.connections)
      plt.imshow(np.uint8(image*255))
      color = ["r", "g", "b",]
      for i, obj in enumerate(conn):
         c = color[i%3]
         for joint1_id, joint2_id in MetaData.pairs:
            x1,y1,v1 = obj[joint1_id]
            x2,y2,v2 = obj[joint2_id]
            if v1*v2 == 0 :
               continue
            plt.plot([x1*4, x2*4], [y1*4, y2*4], f'{c}-')
      plt.show()
      break