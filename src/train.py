import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from data_loader import create_generators
from models.efficientnet import EfficientNet
#from models.cmu import CMU
from utils import set_memory_growth, preprocess_annotations
from callbacks import MapsCompareCallback
from metrics import HeatLoss, PafLoss
from config import Config

# Get the config and update classes.
cfg = Config()
# Set memory growth on gpu's
set_memory_growth()

# Preprocess annotations 
preprocess_annotations(cfg)

# Create generator
train_gen, val_gen = create_generators(cfg)
# Build/Load model 
if cfg.load_checkpoint:
   model = load_model(f'{cfg.checkpoint_load_path}.h5', custom_objects={"HeatLoss": HeatLoss, "PafLoss": PafLoss})
else:
   model = EfficientNet(cfg).build_model()


if cfg.train_base:
   model.trainable = cfg.train_base
model.summary()

model.compile(tf.keras.optimizers.Adam(cfg.learning_rate), 
              loss={"heat_out": HeatLoss(cfg.heat_alpha, cfg.heat_beta),
                    "paf_out": PafLoss(cfg.paf_alpha)}) 

assert model.output_shape == [(None, *cfg.target_maps_size, cfg.n_keypoints), 
                              (None, *cfg.target_maps_size, 2*len(cfg.pairs))], f"Output: {model.output_shape} doesnt fit"

# Create callbacks 
mpc = MapsCompareCallback(model, val_gen, folder=cfg.output_folder)
model_checkpoint = ModelCheckpoint(filepath=cfg.checkpoint_save_path, 
                                    monitor="val_loss",
                                    save_best_only=False, 
                                    save_freq='epoch', 
                                    verbose=1)

model.fit(train_gen,
          validation_data=val_gen,
          epochs=cfg.epochs,
          callbacks=[mpc, model_checkpoint])

