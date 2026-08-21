#!/usr/bin/env python
# coding: utf-8

# # Train 1 UNet/UNeXt/wGAN model on train condition (confluence) and predicting specific target channels.
# Parametrized notebook/script to train under environment variable configured architecture, input, target and confluence settings.
# 
# One run of notebook/script will only train model for one target, confluence combiantion. 
# The notebook is largely for demo purpose on smaller train epoch and batch size setting, and defaults to UNet

# In[4]:


from pathlib import Path
import sys
import os

import numpy as np
import pyarrow.parquet as pq
import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from utils.train_utils import (
    require_env, 
    require_positive_int_env,
    require_bool_env, 
    build_dataset_inputs,
    require_choice_env,
    ARCHITECTURES
)

## Data
from virtual_stain_flow.datasets.crop_dataset import CropImageDataset
from virtual_stain_flow.datasets.base_dataset import BaseImageDataset
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize
from virtual_stain_flow.datasets.ds_engine.crop_generator import generate_point_centered_crops

## Model & Trainer
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.models.unext import ConvNeXtUNet
from virtual_stain_flow.trainers.logging_trainer import SingleGeneratorTrainer
from virtual_stain_flow.models.discriminator import PatchBasedDiscriminator
from virtual_stain_flow.trainers.logging_gan_trainer import LoggingWGANTrainer

## Special losses
from virtual_stain_flow.losses.wgan_losses import (
    AdversarialLoss,
    GradientPenaltyLoss,
    WassersteinLoss
)

## Logging
from virtual_stain_flow.vsf_logging.MlflowLogger import MlflowLogger
from virtual_stain_flow.vsf_logging.callbacks.PlotCallback import PlotPredictionCallback


# ## Training hyper-parameters

# In[ ]:


ON_HPC = require_bool_env("ON_HPC", default=False)
SUBSET_TRAINING = require_bool_env("SUBSET_TRAINING", default=True)

# only needed if ON_HPC
EXPT_NAME = "pediatric_cancer_virtual_stain_training"
SCRATCH_DIR = Path(os.environ.get("SCRATCH", Path.home()))
TRAIN_ROOT = Path(os.environ.get(
    "TRAIN_ROOT",
    '/scratch/alpine/wli19@xsede.org/' if ON_HPC else SCRATCH_DIR / "train_models"
))
TRAIN_ROOT.mkdir(parents=True, exist_ok=True)

# only needed if not ON_HPC
LOCAL_MLFLOW_SERVER = "http://127.0.0.1:5000"

SEED = 42

ARCHITECTURE = require_choice_env(
    "ARCHITECTURE",
    ARCHITECTURES,
    default="UNet",
)

# Default values are used in the notebook; script execution can override them with environment variables.
INPUT_CHANNEL = require_env("INPUT_CHANNEL", default="OrigBrightfield")
TARGET_CHANNEL = require_env("TARGET_CHANNEL", default="OrigDNA")
CONFLUENCE = require_positive_int_env("CONFLUENCE", default=1000)

# Subset mode provides a short local smoke test. Full mode uses the balanced production sample count.
SUBSET_N = 300 if SUBSET_TRAINING else 2_900
EPOCHS = 30 if SUBSET_TRAINING else 300
BATCH_SIZE = 4 if SUBSET_TRAINING else 32
LR = 2e-4
TRAINING_MODE = "subset" if SUBSET_TRAINING else "full"

print(
    "Experiment configuration:\n"
    f"  ARCHITECTURE={ARCHITECTURE}\n"
    f"  TRAINING_MODE={TRAINING_MODE}\n"
    f"  INPUT_CHANNEL={INPUT_CHANNEL}\n"
    f"  TARGET_CHANNEL={TARGET_CHANNEL}\n"
    f"  CONFLUENCE={CONFLUENCE}\n"
    f"  SUBSET_N={SUBSET_N}\n"
    f"  EPOCHS={EPOCHS}\n"
    f"  BATCH_SIZE={BATCH_SIZE}",
    flush=True,
)

LOGGING_TAGS = {
    'run_name': f"{TRAINING_MODE}_{ARCHITECTURE}_{TARGET_CHANNEL}_{CONFLUENCE}_{EPOCHS}",
    'training_mode': TRAINING_MODE,
    'epochs': EPOCHS,
    'confluence': CONFLUENCE,
    'channel': TARGET_CHANNEL,
    'architecture': ARCHITECTURE,
    'batch_size': BATCH_SIZE,
    'lr': LR,
}


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

generator = torch.Generator()
_ = generator.manual_seed(SEED)


# ## Retrieve data split output

# In[4]:


# path resolve for script vs notebook execution
if "__file__" in globals():
    # 2.train_models/nbconverted/2.2.train_unet.py
    TRAIN_EXEC_DIR = Path(__file__).resolve().parents[1]
else:
    # Notebook kernel starts in 2.train_models/
    TRAIN_EXEC_DIR = Path.cwd().resolve()

DATASPLIT_DIR = TRAIN_EXEC_DIR / "data_split_output"
if not DATASPLIT_DIR.exists() or not DATASPLIT_DIR.is_dir():
    raise ValueError(f"Data split output directory {DATASPLIT_DIR} does not exist.")

LOADDATA_FILE_PATH = DATASPLIT_DIR / "loaddata_train.parquet"
if not LOADDATA_FILE_PATH.exists() and not LOADDATA_FILE_PATH.is_file():
    raise ValueError(f"LoadData train file {LOADDATA_FILE_PATH} does not exist.")

LOADDATA_HELDOUT_FILE_PATH = DATASPLIT_DIR / "loaddata_heldout.parquet"
if not LOADDATA_HELDOUT_FILE_PATH.exists() and not LOADDATA_HELDOUT_FILE_PATH.is_file():
    raise ValueError(f"LoadData heldout file {LOADDATA_HELDOUT_FILE_PATH} does not exist.")

SC_FEATURE_FILE = DATASPLIT_DIR / f"sc_profiles.parquet"

if not SC_FEATURE_FILE.exists() and not SC_FEATURE_FILE.is_file():
    raise ValueError(f"Single cell feature file {SC_FEATURE_FILE} does not exist.")

loaddata_train = pq.read_table(LOADDATA_FILE_PATH).to_pandas()

print(f"Initial train loaddata shape: {loaddata_train.shape}")
loaddata_train = loaddata_train.loc[loaddata_train['seeding_density'] == CONFLUENCE]
#loaddata_train = loaddata_train.sample(n=10, random_state=42)
print(f"Filtered train loaddata shape: {loaddata_train.shape}")

loaddata_heldout = pq.read_table(LOADDATA_HELDOUT_FILE_PATH).to_pandas()

print(f"Initial heldout loaddata shape: {loaddata_heldout.shape}")
loaddata_heldout = loaddata_heldout.loc[loaddata_heldout['seeding_density'] == CONFLUENCE]
#loaddata_heldout = loaddata_heldout.sample(n=5, random_state=42)
print(f"Filtered heldout loaddata shape: {loaddata_heldout.shape}")

sc_features = pq.read_table(SC_FEATURE_FILE).to_pandas()


# ## Resolve mlflow tracking

# In[5]:


if ON_HPC:
    # HPC only supports file based tracking that needs to be
    # configured every time, the tracking will resolve to the same
    # directory on scratch space so trainings can be centrally logged and tracked

    EXPT_NAME = "pediatric_cancer_virtual_stain_training"

    TRAIN_ROOT.resolve(strict=False)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_DIR = TRAIN_ROOT / "mlruns"
    TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PLOT_DIR = TRAIN_ROOT / "plots"
    TRAIN_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR = TRAIN_ROOT / 'tmp'
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    mlflow_track_uri = TRAIN_LOG_DIR.resolve().as_uri()
    TRAIN_TRACKING = str(TRAIN_LOG_DIR.resolve())
    mlflow.set_tracking_uri(mlflow_track_uri)
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

    try:
        experiment_id = mlflow.create_experiment(
            name=EXPT_NAME,
            tags={'purpose': 'test'}
        )
        print(f"Created MLflow experiment '{EXPT_NAME}' with ID: {experiment_id}")
    except Exception as e:
        if all(keyword in str(e).lower() for keyword in ['already', 'exists', 'experiment']):
            experiment = mlflow.get_experiment_by_name(EXPT_NAME)
            experiment_id = experiment.experiment_id
            print(f"Experiment '{EXPT_NAME}' already exists with ID: {experiment_id}")
        else:
            print(f"Experiment creation failed: {e}")
            sys.exit(1)

else: # local env with mlflow server running on localhost

    TRAIN_TRACKING = LOCAL_MLFLOW_SERVER


# ## Create datasets

# In[6]:


datasets = {}
datasets_sub = {}

for split, _loaddata_df in zip(
    ["train", "val"],
    [
        loaddata_train,
        loaddata_heldout
    ]
):
    file_index, pt_mapping = build_dataset_inputs(
        _loaddata_df, 
        INPUT_CHANNEL, 
        TARGET_CHANNEL, 
        profile=sc_features
    )

    dataset = BaseImageDataset(
        file_index=file_index.loc[:, [INPUT_CHANNEL, TARGET_CHANNEL]],
        check_exists=True,
    )
    dataset.input_channel_keys = [INPUT_CHANNEL]
    dataset.target_channel_keys = [TARGET_CHANNEL]

    crop_dataset = CropImageDataset.from_base_dataset(
        base_dataset=dataset,
        transforms=MaxScaleNormalize(
            normalization_factor='16bit'),
        how=generate_point_centered_crops,
        crop_size=256,
        mapping=pt_mapping
    )
    num_samples = min(SUBSET_N, len(crop_dataset))
    if num_samples < len(crop_dataset):
        rng = np.random.default_rng(seed= SEED + CONFLUENCE)
        indices = rng.choice(len(crop_dataset), size=num_samples, replace=False)
        datasets_sub[split] = Subset(crop_dataset, indices=indices)
    else:
        datasets_sub[split] = crop_dataset

    datasets[split] = crop_dataset
    print(f"Dataset for split '{split}' has {len(datasets[split])} samples, subset has {len(datasets_sub[split])} samples.")


# ## Configure model and trainer

# In[ ]:


# Batch with DataLoader
# ensure train is reproducibly shuffled
train_loader = DataLoader(datasets_sub['train'], batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0,)
val_loader = DataLoader(datasets_sub['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0,)

## Generator losses shared across architectures
generator_losses = [ # Training with 2 losses: L1 and MS-SSIM
    torch.nn.L1Loss(), # simple per pixel error
    MultiScaleStructuralSimilarityIndexMeasure( # helps models converge much faster
        data_range=None, # use per batch empirical data range
        kernel_size=11, # standard MS-SSIM kernel size, just being explicit
        sigma=1.5, # standard MS-SSIM sigma, just being explicit
    )
]
generator_loss_weights = [1.0, -1.0] # minimize L1 distance (lower is better) and maximize MS-SSIM (higher is better)

# construct model and optimizer, with reproducible initialization
cuda_devices = list(range(torch.cuda.device_count()))
with torch.random.fork_rng(devices=cuda_devices):
    torch.manual_seed(SEED + 1)

    match ARCHITECTURE:
        case "UNet":
            model = UNet(
                in_channels=1,
                out_channels=1,
                depth=4,
                encoder_down_block='conv',
                decoder_up_block='convt',
                act_type='sigmoid'
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            )

            # Initialize Trainer and start training
            trainer = SingleGeneratorTrainer(
                model=model,
                optimizer=optimizer,
                losses=generator_losses,
                loss_weights=generator_loss_weights,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
            )

        case "wGAN":
            model = UNet(
                in_channels=1,
                out_channels=1,
                depth=4,
                encoder_down_block='conv',
                decoder_up_block='convt',
                act_type='sigmoid'
            ).to(device=device,dtype=torch.float32)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
                foreach=False,
                fused=False,
            )
            discriminator = PatchBasedDiscriminator(
                in_channels=2, # real + fake
                base_filters=64,
                n_down_sample_layer = 3,
                n_additional_layer = 1,
                channel_multiplier = 2,
                _leaky_relu_alpha = 0.2,
                _batch_norm = True
            ).to(device,dtype=torch.float32)
            disc_optimizer = torch.optim.AdamW(
                discriminator.parameters(),
                lr=LR,
                betas=(0., 0.999),
                weight_decay=1e-5
            )

            trainer = LoggingWGANTrainer(
                # Generator
                generator=model,
                generator_optimizer=optimizer,
                generator_losses=generator_losses,
                generator_loss_weights=generator_loss_weights,
                # Generator Adverserial Loss
                generator_adverserial_loss=AdversarialLoss(),
                generator_adverserial_loss_weight=0.1,
                # Discriminator wasserstein Loss and Gradient Penalty
                # weights are set to 1.0 and 10.0 respectively, as per the original WGAN-GP paper
                discriminator=discriminator,
                discriminator_optimizer=disc_optimizer,
                discriminator_loss=WassersteinLoss(),
                discriminator_loss_weight=1.0,
                discriminator_gradient_penalty_loss=GradientPenaltyLoss(),
                discriminator_gradient_penalty_weight=10.0,
                # Training alternation
                n_discriminator_steps=3, # number of discriminator updates per generator update
                # Other parameters
                device=device,
                train_loader=train_loader, # training data loader
                val_loader=val_loader,  # validation data loader
                test_loader=None, # for demo purposes, we don't supply a test set
            )

        case "UNeXt":
            model = ConvNeXtUNet(
                in_channels=1,
                out_channels=1,
                decoder_up_block='pixelshuffle',
                decoder_compute_block='convnext',
                act_type = 'sigmoid' # final normalization activation
            ).to(device=device,dtype=torch.float32)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
                foreach=False,
                fused=False,
            )
            # Initialize Trainer and start training
            trainer = SingleGeneratorTrainer(
                model=model,
                optimizer=optimizer,
                losses=generator_losses,
                loss_weights=generator_loss_weights,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=None,
            )

        case _:
            raise AssertionError(
                f"Unhandled architecture: {ARCHITECTURE!r}"
            )


# ## Configure logger

# In[ ]:


# Plotting callback to visualize predictions during training
# At the end of every n epochs, the callback takes the most recent model
# weights and runs inference on the provided images (dataset indexed by sample indices).
# And plots the predictions alongside the inputs and targets to give a visual sense of training progress.
#
# This gets bounded to the logger instance below to automatically register
# plots to the training. 
plot_callback = PlotPredictionCallback(
    name="plot_callback_with_train_data",
    dataset=datasets['train'],
    indices=[0,1,2,3,4], # first 5 samples
    plot_metrics=[torch.nn.L1Loss()],
    every_n_epochs=1, # plot predictions per epoch
    # kwargs passed to plotting backend
    show_plot=False, # don't show plot in notebook
    wspace=0.025, # small spacing between subplots
    hspace=0.05, # small spacing between subplots    
    tag="plot_train_predictions" # tag needed to plot train and val predictions separate
)

plot_callback_val = PlotPredictionCallback(
    name="plot_callback_with_val_data",
    dataset=datasets['val'],
    indices=[0,1,2,3,4], # first 5 samples
    plot_metrics=[torch.nn.L1Loss()],
    every_n_epochs=1, # plot predictions per epoch
    # kwargs passed to plotting backend
    show_plot=False, # don't show plot in notebook
    wspace=0.025, # small spacing between subplots
    hspace=0.05, # small spacing between subplots
    tag="plot_heldout_predictions" # tag needed to plot train and val predictions separate
)

# MLflow Logger
# The logger that communicates with an MLflow tracking server.
# The Mlflow logger by default logs all metrics and losses specified to the
# trainer, plus any files (artifacts in mlflow terminology) generated by the callbacks.
#
# The logger by default saves the model weights at the end of every epoch and
# the best model weights according to validation loss (not applicable here since no val set).
# The only additional callback bound to the logger is plotting callback defined above. 
# Thus the only files being logged are the plots and the model weights.
logger = MlflowLogger(
    name="logger",
    tracking_uri=TRAIN_TRACKING,
    experiment_name=EXPT_NAME,
    run_name=LOGGING_TAGS['run_name'],
    description="/",
    tags={
        key: str(value) for key, value in LOGGING_TAGS.items()
    },
    mlflow_start_run_args={
        'nested': False
    },
    save_model_at_train_end=True,
    save_model_every_n_epochs=1,
    save_best_model=True,
    callbacks=[plot_callback, plot_callback_val],
)


# In[8]:


trainer.train(logger=logger, epochs=EPOCHS)
logger.end_run()
