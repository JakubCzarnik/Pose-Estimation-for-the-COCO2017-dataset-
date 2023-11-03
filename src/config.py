from data_loader import MetaData, DataGenerator
from utils import MapsParser

class Config:
   def __init__(self):
      ##### Paths #####
      self.train_annotations = "D:/COCO 2017/annotations/instances_train2017.json"  # Path to original train annotations
      self.val_annotations = "D:/COCO 2017/annotations/instances_val2017.json"      # Path to original val annotations
      
      self.extracted_train_annotations = "train_annotations.json" # Path to extracted train annotations from original train annotations
      self.extracted_val_annotations = "val_annotations.json"     # Path to extracted val annotations from original val annotations
      
      self.train_images = "D:/COCO 2017/train2017/"   # Folder path to train images
      self.val_images = "D:/COCO 2017/val2017/"       # Folder path to val images

      self.checkpoint_load_path = "checkpoints/last"  # Model checkpoint path to load
      self.load_checkpoint = False                     # load checkpoint from {checkpoint_load_path}?
      self.train_base = False                         # unlock all layers for training? (If some was frozen)
      
      ##### General #####
      self.target_image_size = (256, 256) # model input size in format: (h, w)
      self.target_maps_size = (64, 64)    # model output size: (h, w)
      
      self.epochs = 100                   # Num of epochs till end the training
      self.batch_size = 4                 # Num of samples in single batch
      self.train_batches_per_epoch = 1000 # On every epoch will be choosen {} batches from whole train dataset
      self.val_batches_per_epoch = 1      # On every epoch will be choosen {} batches from whole val dataset
      self.learning_rate = 5e-4           # Optimizer learning rate

      ##### MetaData ##### 
      self.sigma = 1.3        # Coefficient for function blurring keypoints 
      self.vector_width = 4   # Vector width in Part Affinity Fields (PAFs)
      self.n_keypoints = 18   # 17 + 1 additional (middle of arms)
      
      self.pairs = [(17,0),                                  # head-body     
         (0, 1), (0, 2), (1, 3), (2, 4),                     # head
         (17, 5),  (17, 6), (5, 7), (7, 9), (6, 8), (8, 10), # body
         (11, 17), (17, 12),                                 # body-legs
         (12, 11), (11, 13), (12, 14), (13, 15), (14, 16)    # legs
      ]                                                      # The order matters for merging 

      ##### Model #####
      self.l1 = 0             # l1 regularizer coefficient
      self.l2 = 0             # l2 regularizer coefficient
      self.leaky_alpha = 0.2  # LeakyReLU alpha coefficient
      self.bn_momentum = 0.9  # BatchNormalization momentum

      ##### Metrics #####
      self.heat_alpha = 2.8 # The weights where there is a point will be: e^(alpha) instead of 1.
      self.heat_beta = 4    # The final loss will be multiplied by 5 (to balance between the PAFs loss).
      self.paf_alpha = 2.5  # The weights where there is a vector will be: e^(alpha) instead of 1.

      ##### MapParser #####
      self.maxima_filter_size = 3   # Keypoint are selected from every 3x3 subset of pixels in heatmap
                                    # where the given keypoint has the greatest value amoung others.
      self.heatmaps_threshold = 0.8 # Minimum value of maxima to pass the point.
      self.pafs_threshold = 0.75    # Minimum score of connection between points to pass the connection.
      self.conections_threshold = 4 # Minimum connected connections in a single instance to pass the instance, 
                                    # ex. if person has fewer than 4 connections they will not be considered.

      ##### Callbacks #####
      self.output_folder = "training_output/"            # Folder to save results from model on validation data.
      self.checkpoint_save_path = "checkpoints/last.h5"  # Model Checkpoint save path

      ##### Update Classes #####
      self.update_settings()


   def update_settings(self):
      DataGenerator.update(self)
      MapsParser.update(self)
      MetaData.update(self)