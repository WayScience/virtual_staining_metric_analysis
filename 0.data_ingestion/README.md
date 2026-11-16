# PedCP90 data ingestion

Prepare the PedCP90 reference dataset for downstream analyses: download source metadata and image-quality measurements, exclude incomplete or QC-failing fields, select annotated 24-hour images, and collect images with cell and nuclear masks.
Summary plots and representative previews support inspection of the resulting data.

## Configure

Activate the project's Python environment and authenticate/mount the institutional data before running preparation or ingestion. 

Edit [project_config.yaml](../project_config.yaml):

- `metadata_root`: downloaded metadata and the prepared image file index;
  defaults to the repository's metadata directory.
- `large_output_root`: destination for the reference image collection.
- `image_root`: local root of the source microscopy images.
- `outline_root`: directory containing the per-plate cell and nuclear outlines.
- `workers`: parallel ingestion workers.
- `preview_count` and `preview_seed`: number of preview FOVs and reproducible selection.

## Running

From the repository root, run the **entire workflow** with:

```bash
bash 0.data_ingestion/run.sh
```

Or run individual stages:

```bash
bash 0.data_ingestion/run.sh download # download metadata only
bash 0.data_ingestion/run.sh prepare # resolve to local paths, apply qc filtering
bash 0.data_ingestion/run.sh ingest --dry-run  # Optional preflight; writes nothing
bash 0.data_ingestion/run.sh ingest # write the ome-arrow outputs
bash 0.data_ingestion/run.sh report # ingestion & qc reports
```

## Outputs

| Output | Location |
| --- | --- |
| Original barcode mapping, platemaps, LoadData and QC CSVs | `<metadata_root>/pedcp90/raw/<source_commit>/` |
| QC-filtered, annotated 24-hour image file index (Parquet) | `<metadata_root>/pedcp90/loaddata_for_degradation.parquet` |
| Reference images, cell/nuclear masks and associated tables | `<large_output_root>/pedcp90/reference/` |
| QC before/after counts | `0.data_ingestion/results/qc_retention.png` |
| Field counts through preparation | `0.data_ingestion/results/preparation_retention.png` |
| Field counts before/after outline selection and ingestion | `0.data_ingestion/results/outline_retention.png` |
| Representative channel and mask previews | `0.data_ingestion/results/montage_*.png` |

Plot locations are relative to the repository, regardless of the working directory.
`<source_commit>` identifies the pinned version of the upstream [PCCMA profiling repository](https://github.com/WayScience/pediatric_cancer_atlas_profiling) used for the downloads. 
Downloaded originals are retained unchanged.
