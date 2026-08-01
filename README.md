# Analysis of full reference image quality assessment metrics in context of assessing virtual staining model performance and generalizability

This repository houses the code for the analysis used to characterize image-quality assessment metrics and evaluate virtual staining models for label-free microscopy.

## Motivation & Goals

In developing or assessing virtual staining methods for high-content imaging, models are often applied across changes in cell line, cell density, imaging conditions, and other biological or technical variables.

Most full-reference image-quality assessment metrics used in current virtual staining work were originally developed for natural images and are not guaranteed to be content invariant.
In the worst case scenairo, this implies that the same reconstruction error may produce different metric values when the underlying biological image content changes.

This repository is therefore about examining metric behavior in context of variable image content.
The objective is approached in two stages:
1. Metrics are exclusively analyzed against controlled image degradation and known data-inherent biological covariates.
This allows us to study the behavior of metrics against simple and well-understoood forms of image quality degradation.

2. We train real virtual staining models on the same dataset, and generate model predictions in unseen cell lines and/or plating conditions, compute metrics comparing model predictions and the ground truth image, and analyze realistic metric behavior under biological distribution shift.
