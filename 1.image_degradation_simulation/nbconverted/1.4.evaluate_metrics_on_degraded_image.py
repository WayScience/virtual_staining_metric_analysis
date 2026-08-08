#!/usr/bin/env python
# coding: utf-8

# # Evaluates metrics on previously generated reference degraded image pairs
# Iterates over images in lance as mini batches, reads one fragment only once as that is expensive. 

# In[1]:


import pandas as pd
import torch
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from utils.validate_config import (
    load_yaml_config,
    require_config_directory,
)
from utils.custom_metrics import (
    ForegroundPSNR,
    ForegroundSSIM,
    ReusableDISTS,
    ReusableLPIPS,
)
from utils.metric_spec import MetricSpec
from utils.pair_dataset import PairedLanceImageDataset, collate_paired_lance_samples
from utils.eval_write_metric import evaluate_lance_metrics_to_parquet


# In[2]:


config = load_yaml_config("degradation_config.yaml")
analysis_dir = require_config_directory(config, "analysis_out_dir")

reference_patch_dir = analysis_dir / "patches" / "reference_records"
if not reference_patch_dir.exists():
    raise RuntimeError(
        f"Reference patch directory {reference_patch_dir} does not exist. "
        "Run notebook 1.2 first."
    )

reference_lance_dir = reference_patch_dir / "data.lance"
if not reference_lance_dir.exists():
    raise RuntimeError(
        f"Reference Lance directory {reference_lance_dir} does not exist. "
        "Run notebook 1.2 first."
    )

degrade_patch_dir = analysis_dir / "patches" / "degraded_records"
if not degrade_patch_dir.exists():
    raise RuntimeError(
        f"Degraded patch directory {degrade_patch_dir} does not exist. "
        "Run notebook 1.3 first."
    )

degrade_lance_dir = degrade_patch_dir / "data.lance"
if not degrade_lance_dir.exists():
    raise RuntimeError(
        f"Degraded Lance directory {degrade_lance_dir} does not exist. "
        "Run notebook 1.3 first."
    )

output_dir = analysis_dir / "patches" / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


degradation_catalog_file = analysis_dir / "degradation_catalog.csv"
if not degradation_catalog_file.exists():
    raise FileNotFoundError(
        f"Degradation catalog file {degradation_catalog_file} does not exist. "
        "Run notebook 1.3 first."
    )
degradation_catalog = pd.read_csv(degradation_catalog_file)


# In[5]:


metric_specs = {
    "ssim": MetricSpec(
        name="ssim",
        metric=structural_similarity_index_measure,
        kwargs={"data_range": 1.0, "reduction": "none"},
    ),
    "psnr": MetricSpec(
        name="psnr",
        metric=peak_signal_noise_ratio,
        kwargs={"data_range": 1.0, "reduction": "none", "dim": (1, 2, 3)},
    ),
    "mae": MetricSpec(
        name="mae",
        metric=l1_loss,
        kwargs={"reduction": "none"},
    ),
    "lpips": MetricSpec(
        name="lpips",
        metric=ReusableLPIPS(normalize=True),
        kwargs={"reduction": "none"},
        input_channels=3,
    ),
    "dists": MetricSpec(
        name="dists",
        metric=ReusableDISTS(),
        kwargs={"reduction": "none"},
        input_channels=3,
    ),
    # foreground variants restrict the metric to the pixels selected by a
    # per-image Otsu threshold of the reference image
    "foreground_ssim": MetricSpec(
        name="foreground_ssim",
        metric=ForegroundSSIM(data_range=(0.0, 1.0)),
        kwargs={"reduction": "none"},
    ),
    "foreground_psnr": MetricSpec(
        name="foreground_psnr",
        metric=ForegroundPSNR(data_range=(0.0, 1.0)),
        kwargs={"reduction": "none"},
    ),
}


# In[6]:


dataset = PairedLanceImageDataset(
    reference_uri=reference_lance_dir,
    degradation_uri=degrade_lance_dir,
    pair_key="record_id",
    scan_batch_size=64,
    batch_readahead=2,
    validate_pairing=True,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    collate_fn=collate_paired_lance_samples,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    multiprocessing_context="spawn",
    drop_last=False,
)


# In[7]:


metric_output_dirs = evaluate_lance_metrics_to_parquet(
    metric_specs,
    loader,
    degradation_catalog,
    output_dir,
    overwrite=False,
    device=device,
)
