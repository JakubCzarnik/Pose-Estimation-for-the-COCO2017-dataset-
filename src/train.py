import tensorflow as tf
from data_loader import *
from model import *
from utils import extract_coco
from callbacks import MapsCompareCallback
from metrics import ExponentialMSE
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) == 0:
   raise SystemError
for gpu in gpus:
   print(gpu) 
   tf.config.experimental.set_memory_growth(gpu, True)


### Settings ###
MetaData.target_image_size = (256, 256)
MetaData.target_maps_size = (64, 64)
MetaData.sigma = 1
MetaData.vec_width = 4

DataGenerator.batch_size = 5
train_batches = 2500
val_batches = 150

lr = 5e-6
epochs = 150

load_checkpoint = False
checkpoint_name = "checkpoints/last"
annotations_folder = "D:/COCO 2017/annotations/"
train_dataset = "D:/COCO 2017/train2017/"
val_dataset = "D:/COCO 2017/val2017/"


### Preprocess annotations ###
if not os.path.isfile("train_annotations.json"):
   extract_coco(f"{annotations_folder}instances_train2017.json", save_filename="train_annotations.json")
if not os.path.isfile("val_annotations.json"):
   extract_coco(f"{annotations_folder}instances_val2017.json", save_filename="val_annotations.json")


train_gen = DataGenerator("train_annotations.json", 
                          train_dataset, 
                          batches=train_batches)
val_gen = DataGenerator("val_annotations.json", 
                        val_dataset, 
                        batches=val_batches)


if load_checkpoint:
   model = load_model(f'{checkpoint_name}.h5', custom_objects={"ExponentialMSE": ExponentialMSE})
else:
   model = build_model(MetaData.n_keypoints, 2*len(MetaData.pairs))
model.summary()

model.compile(tf.keras.optimizers.Adam(lr), 
              loss={"heat_out": ExponentialMSE(alpha=7, beta=0.01),
                    "paf_out": ExponentialMSE(alpha=6, beta=0.01)})


# callbacks
mpc = MapsCompareCallback(model, val_gen, folder="training_output/")
model_checkpoint = ModelCheckpoint(filepath="checkpoints\last.h5", 
                                 monitor="val_loss",
                                 save_best_only=True, 
                                 save_freq='epoch', 
                                 verbose=1)

model.fit(train_gen,
          validation_data=val_gen,
          epochs=200,
          callbacks=[mpc, model_checkpoint])





