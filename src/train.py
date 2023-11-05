import tensorflow as tf
from utils import set_memory_growth, preprocess_annotations
from data_loader import create_generators
from models.efficientnet import EfficientNet
from models.pose import PoseDetector
from metrics import HeatLoss, PafLoss
from callbacks import MapsCompareCallback, Checkpoint, TensorBoard, ProgressBar
from config import Config

# Get the config and update classes.
cfg = Config(show_config=True)

# Set memory growth on gpu's
set_memory_growth()

# Preprocess annotations 
preprocess_annotations(cfg)

# Create generators
train_gen, val_gen = create_generators(cfg)

# Build model 
model = EfficientNet(cfg).build_model()


heat_loss = HeatLoss(cfg.heat_alpha, cfg.heat_beta)
paf_loss = PafLoss(cfg.paf_alpha)

opt = tf.keras.optimizers.Adam(cfg.learning_rate)


detector = PoseDetector(model, cfg)
detector.compile(opt, heat_loss, paf_loss)

# Create callbacks 
checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
manager = tf.train.CheckpointManager(checkpoint, cfg.checkpoint_save_path, max_to_keep=cfg.checkpoints_to_keep)
callbacks = []

model_checkpoint = Checkpoint(manager, save_best_only=False)
mpc = MapsCompareCallback(detector, val_gen, folder=cfg.output_folder)
bar = ProgressBar(metrics=cfg.metrics_to_print)

callbacks.append(bar)
callbacks.append(model_checkpoint)
callbacks.append(mpc)

if cfg.run_tensorboard:
   import datetime
   folder_name = datetime.datetime.now().strftime("%d_%H-%M")
   tb = TensorBoard(log_dir=f"logs/{folder_name}")
   callbacks.append(tb)

# load checkpoint
checkpoint.restore(tf.train.latest_checkpoint(cfg.output_folder)) if cfg.load_last_checkpoint else None
checkpoint.restore(cfg.checkpoint_load_path) if cfg.load_checkpoint else None

# unlock layers
model.trainable = cfg.unlock_layers if cfg.unlock_layers else model.trainable
 
# train   
detector.fit(train_gen, validation_data=val_gen, epochs=cfg.epochs, callbacks=callbacks, verbose=0)
