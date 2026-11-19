# PedCP90 data ingestion

Prepare the unchanged annotated 24-hour PedCP90 reference **and five density-specific
U2-OS training candidate pools** by default. These are candidate pools, not finalized
train/validation splits: there is no balancing, downsampling or crop export.
Summary plots and representative previews support inspection of each collection.

Training uses `Assay_Plate1_platemap`, U2-OS, seeding densities **1000, 2000, 4000,
8000 and 12000**, and all three timepoints. “Plate 1” means this platemap across
three acquisition barcodes, not only the first barcode:

| Timepoint (hours) | Acquisition barcode |
| --- | --- |
| 24 | BR00143976 |
| 48 | BR00143978 |
| 72 | BR00143980 |

All outputs use the same QC statistics pooled over the full six-plate source pool
before TIFF completeness/cohort selection, the same plate-specific illumination
correction (fluorescence subtraction, brightfield division), and the same cell/nuclear
outline-to-mask reconstruction. Each density export retains biological, well and
timepoint metadata, including `platemap_file`; the legacy reference schema, ordering,
dataset identity and paths remain unchanged (no `platemap_file` column).

> **Source overlap:** the reference includes 24-hour plate-1 U2-OS fields also eligible
> for training. Distinct dataset/FOV IDs do **not** make these acquisitions independent.
> Before future evaluation, exclude overlapping source `(plate, well, site)` keys and
> assign well-group splits before generating crops. Reference membership is not changed
> by this workflow.

## Configure

Activate the project's Python environment and authenticate/mount the institutional data before running preparation or ingestion. 

Edit [project_config.yaml](../project_config.yaml):

- `metadata_root`: downloaded metadata and the prepared image file index;
  defaults to the repository's metadata directory.
- `large_output_root`: destination for the reference and five training image collections.
- `image_root`: local root of the source microscopy images.
- `outline_root`: directory containing the per-plate cell and nuclear outlines.
- `workers`: parallel ingestion workers.
- `preview_count` and `preview_seed`: number of preview FOVs and reproducible selection.

## Running

From the repository root, run the **entire workflow** (reference plus all five
training pools) with:

```bash
bash 0.data_ingestion/run.sh
```

Or run individual stages:

```bash
bash 0.data_ingestion/run.sh download # download metadata only
bash 0.data_ingestion/run.sh prepare # resolve paths, apply shared QC, select all pools
bash 0.data_ingestion/run.sh ingest --dry-run  # Optional preflight; writes nothing
bash 0.data_ingestion/run.sh ingest # write the ome-arrow outputs
bash 0.data_ingestion/run.sh report # ingestion & qc reports
```

Use the optional `--cohort all|reference|training` selector to process only the
desired collections (`all` is the default):

```bash
bash 0.data_ingestion/run.sh run --cohort reference
bash 0.data_ingestion/run.sh prepare --cohort training
bash 0.data_ingestion/run.sh ingest --cohort training --dry-run
bash 0.data_ingestion/run.sh ingest --cohort training
bash 0.data_ingestion/run.sh report --cohort training
```

The selector applies to `prepare`, `ingest`, `report` and `run`; `download` always
uses the shared pinned source cache and ignores it. `--dry-run` is supported only
for `ingest`, requires prepared indexes, checks every selected collection and writes
no indexes, collections or figures. Normal ingestion resumes unchanged FOVs in each
independent collection. Reports read exported collections without source image or
outline mounts and do not require unselected collections.

Preparation requires a nonempty QC-passing selection at every requested training
density and all three timepoints; it validates all requested selections before
writing any index. Missing outlines are skipped by the existing backend. Ingestion
fails if a density has no eligible FOVs and warns if outline availability removes
an entire timepoint; inspect prepared-versus-written coverage rather than assuming
each candidate pool remains complete.

## Outputs

| Output | Location |
| --- | --- |
| Original barcode mapping, platemaps, LoadData and QC CSVs | `<metadata_root>/pedcp90/raw/<source_commit>/` |
| QC-filtered, annotated 24-hour image file index (Parquet) | `<metadata_root>/pedcp90/loaddata_for_degradation.parquet` |
| Reference images, cell/nuclear masks and associated tables | `<large_output_root>/pedcp90/reference/` |
| Training file index, one per density | `<metadata_root>/pedcp90/training/density_<N>/loaddata.parquet` |
| Independent training images, masks and associated tables | `<large_output_root>/pedcp90/training/density_<N>/` |
| Per-density preparation/outline retention and previews | `0.data_ingestion/results/training/density_<N>/` |
| Shared pooled-source QC before/after counts (all cohorts) | `0.data_ingestion/results/qc_retention.png` |
| Field counts through preparation | `0.data_ingestion/results/preparation_retention.png` |
| Field counts before/after outline selection and ingestion | `0.data_ingestion/results/outline_retention.png` |
| Representative channel and mask previews | `0.data_ingestion/results/montage_*.png` |

Plot locations are relative to the repository, regardless of the working directory.
Reference and training preparation plots both show Raw → TIFF complete → QC passed →
cohort selection → Annotated counts for each acquisition plate, with imaging timepoints
in the legend. The training cohort filter includes platemap, cell line, seeding density
and requested timepoints; its drop is intentional selection, not additional QC failure.
The pooled-source QC plot is saved only once at the shared results root, including for
training-only preparation. Re-preparing a density removes its obsolete duplicate QC plot.

`<source_commit>` identifies the pinned version of the upstream [PCCMA profiling repository](https://github.com/WayScience/pediatric_cancer_atlas_profiling) used for the downloads. 
Downloaded originals are retained unchanged.

`<N>` is one of the five densities above. Training collections are self-contained,
not links to the reference or each other; overlapping reference pixels are deliberately
exported again.

## Future well splits and nucleus-centered crops

The existing collection interface supports filtering metadata by acquisition well,
reading six aligned intensity planes plus cell/nuclear label planes, and using object
tables (centroids, bounding boxes, labels and nucleus-cell relationships) for spatially
aligned nucleus-centered slices. This is readiness for a future consumer, not a crop
or Torch wrapper in this workflow.

That consumer must define well-group train/validation assignment, channel order,
normalization, edge padding/exclusion and **bounded per-worker FOV caching** before
training. Full planes are loaded by `read_fov`; repeatedly reopening them for every
nucleus is not an efficient crop loader. Group at least by `(plate, well)`, not by
individual sites or crops. Label/object IDs describe reconstructed outline interiors,
**not original CellProfiler object IDs**; do not assume original single-cell profile
identities or centroids are preserved.
