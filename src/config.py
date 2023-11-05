from data_loader import MetaData, DataGenerator
from utils import MapsParser

class Config:
   def __init__(self, show_config=False):
      ##### Paths #####
      self.train_annotations = "D:/COCO 2017/annotations/instances_train2017.json"  # Path to original train annotations
      self.val_annotations = "D:/COCO 2017/annotations/instances_val2017.json"      # Path to original val annotations
      
      self.extracted_train_annotations = "train_annotations.json" # Path to extracted train annotations from original train annotations
      self.extracted_val_annotations = "val_annotations.json"     # Path to extracted val annotations from original val annotations
      
      self.train_images = "D:/COCO 2017/train2017/"   # Folder path to train images
      self.val_images = "D:/COCO 2017/val2017/"       # Folder path to val images


      self.checkpoint_load_path = "checkpoints/last"  # Model checkpoint path to load
      self.load_checkpoint = False                    # will load a checkpoint from the {checkpoint_load_path}
      self.load_last_checkpoint = False               # will load last saved checkpoint
      self.unlock_layers = False                      # unlock all layers for training? (If some was frozen)
      
      assert not (self.load_checkpoint and self.load_last_checkpoint), "Choose one way to load the checkpoint"

      ##### General #####
      self.target_image_size = (256, 256) # model input size in format: (h, w)
      self.target_maps_size = (64, 64)    # model output size: (h, w)
      
      self.epochs = 100                   # Num of epochs till end the training
      self.batch_size = 4                 # Num of samples in single batch
      self.train_batches_per_epoch = 1000 # On every epoch will be choosen {} batches from whole train dataset
      self.val_batches_per_epoch = 1      # On every epoch will be choosen {} batches from whole val dataset
      self.learning_rate = 1e-3           # Optimizer learning rate

      ##### MetaData ##### 
      self.sigma = 0.8        # Coefficient for function blurring keypoints 
      self.vector_width = 4   # Vector width in Part Affinity Fields (PAFs)
      self.n_keypoints = 18   # 17 + 1 additional (point between arms)
      
      self.pairs = [(17,0),                                  # head-body     
         (0, 1), (0, 2), (1, 3), (2, 4),                     # head
         (17, 5),  (17, 6), (5, 7), (7, 9), (6, 8), (8, 10), # body
         (11, 17), (17, 12),                                 # body-legs
         (12, 11), (11, 13), (12, 14), (13, 15), (14, 16)    # legs
      ]                                                      # *The order matters for merging* 

      ##### Model #####
      self.l1 = 1e-5             # l1 regularizer coefficient (heatmaps layers)
      self.l2 = 1e-5             # l2 regularizer coefficient (pafs + extraction layers)
      self.leaky_alpha = 0.1  # LeakyReLU alpha coefficient
      self.bn_momentum = 0.9  # BatchNormalization momentum

      ##### Metrics #####
      self.heat_alpha = 3.2   # The weights where there is a point will be: e^(alpha) instead of 1.
      self.heat_beta = 4      # The final loss will be multiplied by 5 (to balance between the PAFs loss).
      self.paf_alpha = 2.5    # The weights where there is a vector will be: e^(alpha) instead of 1.
      
      self.stage_weights = [1, 1.5, 2] # Every stage loss will be multiplicated by his own value

      ##### MapParser #####
      self.maxima_filter_size = 3   # Keypoint are selected from every 3x3 subset of pixels in heatmap
                                    # where the given keypoint has the greatest value amoung others.
      self.heatmaps_threshold = 0.8 # Minimum value of maxima to pass the point.
      self.pafs_threshold = 0.75    # Minimum score of connection between points to pass the connection.
      self.conections_threshold = 4 # Minimum connected connections in a single instance to pass the instance, 
                                    # ex. if person has fewer than 4 connections they will not be considered.

      ##### Callbacks #####
      self.output_folder = "training_output/"      # Folder to save results from model on validation data.
      self.checkpoint_save_path = "checkpoints/"   # Model Checkpoint save path
      self.checkpoints_to_keep = 1                 # how much checkpoints keep in folder
      self.run_tensorboard = True                  # will run the TensorBoard callback

      self.metrics_to_print = ["loss", "heat_2", "paf_2", "val_loss"] # metrics to print in the console

      ##### Update Classes #####
      self.update_settings()

      if show_config:
         self.show_config()


   def update_settings(self):
      DataGenerator.update(self)
      MapsParser.update(self)
      MetaData.update(self)


   def show_config(self):
      print("|--------------------------------------|")
      print("|                                      |")
      print("|               Settings               |")
      print("|                                      |")
      print("|--------------------------------------|")
      print(f"| {self.target_image_size = }")
      print(f"| {self.target_maps_size = }")
      print(f"|")
      print(f"| {self.epochs = }")
      print(f"| {self.batch_size = }")
      print(f"| {self.train_batches_per_epoch = }")
      print(f"| {self.val_batches_per_epoch = }")
      print(f"| {self.learning_rate = }")
      print(f"|")
      print(f"| {self.sigma = }")
      print(f"| {self.vector_width = }")
      print(f"|")
      print(f"| {self.heat_alpha = }")
      print(f"| {self.heat_beta = }")
      print(f"| {self.paf_alpha = }")
      print(f"| {self.stage_weights = }")
      print(f"|")
      print(f"|               LOAD CHECKPOINT")
      print(f"| {self.checkpoint_load_path = }")
      print(f"| {self.load_checkpoint = }")
      print(f"| {self.load_last_checkpoint = }")
      print("|--------------------------------------|")
