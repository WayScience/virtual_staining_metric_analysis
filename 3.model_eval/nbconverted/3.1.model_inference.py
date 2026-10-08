#!/usr/bin/env python
# coding: utf-8

# # 3.1. Generate model predictions iwth brightfield inputs
# 
# This notebook loads trained models and generates predictions using input crops written in 3.0.
# 
# Outputs are written in the same format as the inputs grouping related predictions (from the same input) together as a row entry.

# In[1]:


from pathlib import Path
import json
import importlib
from itertools import product

import torch
import pandas as pd
from tqdm.auto import tqdm

from utils.validate_config import (
    load_yaml_config,
    require_config_directory,
)
from utils.write_model_inference import write_model_predictions


# In[2]:


degradation_path = Path(".").resolve().parent / "1.image_degradation_simulation"
if not degradation_path.exists():
    raise FileNotFoundError(f"Degradation simulation directory not found: {degradation_path}")

config = load_yaml_config(degradation_path / "degradation_config.yaml")
analysis_dir = require_config_directory(config, "analysis_out_dir")

br_input_dir = analysis_dir / "patches" / "model_inference_inputs" / "OrigBrightfield"

if not br_input_dir.exists():
    raise RuntimeError(
        f"BR input directory {br_input_dir} does not exist. "
        "Run notebook 3.0 first."
    )

br_input_lance_dir = br_input_dir / "data.lance"
if not br_input_lance_dir.exists():
    raise RuntimeError(
        f"BR input Lance directory {br_input_lance_dir} does not exist. "
        "Run notebook 3.0 first."
    )

pred_out_dir = analysis_dir / "patches" / "model_predictions"
pred_out_dir.mkdir(parents=True, exist_ok=True)

# modify to where models downloaded/transferred local
LOGGING_ROOT = Path("/mnt/hdd20tb2/alsf_r2_transfer/tracking")
if not LOGGING_ROOT.exists():
    raise FileNotFoundError(f"Logging root directory not found: {LOGGING_ROOT}")


# ## Read in and validate all models
# Models are trained on HPC with legacy mlflow file based tracking and transferred to local disk.
# Accessing models with mlflow requires starting a local file based server external of the notebook, which adds complexity. 
# So here we adpot the approach of manually discovering trained model files to go around external server requirement.

# In[3]:


# Assumes flat directory structure under LOGGING_ROOT for runs
# which will be the case if models are trained and logged with mlflow file-based backend
runs = LOGGING_ROOT.iterdir()

validated_runs = []

for run in runs:

    if not run.is_dir():
        continue

    # basic mlflow logged run checks
    metadata_file = run / "meta.yaml"
    if not metadata_file.exists():
        continue 
    tag_dir = run / "tags"
    if not tag_dir.exists():
        continue

    # access stored tags for channel, density (confluence), and model architecture
    tag_files = [file for file in tag_dir.glob("*") if file.is_file()]
    tags = {
        tag_file.name: tag_file.read_text(encoding="utf-8").strip()
        for tag_file in tag_files
    }
    target_channel = tags.get("channel", None)
    train_density = tags.get("confluence", None)
    model = tags.get("model.0.class_path", None)
    train_architecture = tags.get("run_name", None)
    if target_channel is None or train_density is None or model is None or train_architecture is None:
        continue

    train_density = int(train_density)
    train_architecture = train_architecture.split("_")[1]

    # attempt to locate the best weight file based on validation L1 loss
    weight_artifact_dir = run / "artifacts" / "weights"
    if not weight_artifact_dir.exists():
        continue

    weight_files = list(weight_artifact_dir.glob("*.pth"))
    if not any(weight_files):
        continue

    metric_dir = run / "metrics"
    if not metric_dir.exists():
        continue

    val_l1_file = metric_dir / "val_L1Loss"
    if not val_l1_file.exists() or not val_l1_file.is_file():
        continue

    val_l1 = pd.read_csv(val_l1_file, sep=r'\s+', header=None)
    best_val_epoch = val_l1[1].idxmin()

    best_weight = [
        w for w in weight_files 
        if w.stem.endswith(f"_{best_val_epoch + 1}")
    ]
    if not best_weight:
        continue

    # this should not happen, but we handle it just in case of corrupted/interrupted transfer
    if len(best_weight) > 1:
        print(f"Warning: Multiple weight files found for best epoch {best_val_epoch + 1} in run {run}. Using the first one.")

    # discover the model architecture class path and config
    model_config = run / "artifacts" / "configs" / f"{model.split(".")[-1]}.json"
    if not model_config.exists():
        continue

    with open(model_config, 'r') as f:
        model_config_data = json.load(f)

    validated_runs.append({
        "run_path": run,
        "train_architecture": train_architecture,
        "target_channel": target_channel,
        "train_density": train_density,
        "model": model,
        "config": model_config_data,
        "best_weight": best_weight[0],
    })

print(f"Found {len(validated_runs)} validated runs.")


# ### Next we organize the models by their architecture and density and load all model weights into RAM.
# 
# This helps with generating a prediction catalogue that defines the exact order of 75 model predictions over the same brightfield input.
# 
# The action of loading everything into RAM may seem demanding but because all models here are convolutional neural networks and quite lightweight all 75 occupies roughly 13GB of RAM. 
# This earlier memory usage helps eliminate the need for repeated model reads from disk during prediction.

# In[4]:


target_channels = sorted(
    set(run["target_channel"] for run in validated_runs)
)
architectures = sorted(
    set(run["train_architecture"] for run in validated_runs)
)
densities = sorted(
    set(run["train_density"] for run in validated_runs)
)

sequence = list(product(
    architectures,
    densities,
))
reverse_lookup = {
    (arch, dens): i
    for i, (arch, dens) in enumerate(sequence)
}
model_catalogue = pd.DataFrame(
    sequence,
    columns=["architecture", "density"],
)

channel_to_models = {
    chan: [None] * len(model_catalogue)
    for chan in target_channels
}

# While organizing the models by channel, architecture, and density, load them
for run in tqdm(validated_runs, total=len(validated_runs), desc="Loading models"):

    channel = run["target_channel"]
    architecture = run["train_architecture"]
    density = run["train_density"]
    i = reverse_lookup[(architecture, density)]

    model_class_path = run["model"]
    module_name, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    if class_name == "ConvNeXtUNet":
        run["config"]["init"] = {**run["config"]["init"], "_pixel_shuffle_preserve_channels": True}

    model_instance = model_class(**run["config"]["init"])
    model_instance.load_state_dict(torch.load(run["best_weight"], map_location=torch.device('cpu')))
    model_instance.eval()

    channel_to_models[channel][i] = model_instance

if any(model is None for models in channel_to_models.values() for model in models):
    print("Warning: Some models are missing for certain architecture-density combinations.")

model_catalogue.head()
model_catalogue.to_csv(pred_out_dir / "model_catalogue.csv", index=False)


# ## Prediction configuration
# Adjust to fit on hardware

# In[5]:


batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} with batch size {batch_size}")


# In[6]:


for chan in target_channels:

    model_list = channel_to_models[chan]

    print(chan)

    input_dir = analysis_dir / "patches" / "model_inference_inputs" / "OrigBrightfield"
    if not input_dir.exists():
        raise RuntimeError(
            f"BR input directory {input_dir} does not exist. "
            "Run notebook 3.0 first."
        )
    # not needed for inference step, needed in next notebook for computing metrics
    # target_dir = analysis_dir / "patches" / f"model_inference_{chan}_target"

    _output_dir = pred_out_dir / chan
    _output_dir.mkdir(parents=True, exist_ok=True)

    write_model_predictions(
        models=model_list,
        model_catalogue=model_catalogue,
        output_dir=_output_dir,
        reference_lance_dir=input_dir / "data.lance",
        device=device,
        batch_size=batch_size,
    )
