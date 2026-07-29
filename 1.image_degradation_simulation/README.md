# Image quality assessment metrics vs. simulated image degradation

## Goals

This simulation analysis is about the sensitivity and susceptibility to confounding of common image quality assessment metrics in context of label-free microscopy virtual staining.

## Dataset
We use a Cell Painting dataset consisting of 18 unperturbed pediatric cancer cell lines plated under 5 different levels of seeding density (https://github.com/WayScience/pediatric_cancer_atlas_profiling):
- `1.0.get_metadata.ipynb`
- `1.1.preprocess_data.ipynb`
- `1.2.write_image_patches.ipynb`

## Approach

### Image degrading transforms
We first selected a suite of 6 families degrading image transforms to simulate artifacts from virtual staining models generating imperfect predictions:
- Gaussian noise (additive)
- Gaussian blur
- Erosion (grayscale morpholical operation)
- Dilation (grayscale morpholical operation)
- Gamma correction (dimming and brightening)
- Grid distortion
And apply them to the fluoresence Cell Painting channels of our dataset in 6 incremental levels of severity by tuning a single transformation parameter while leaving others fixed (see code for details).

This produces 36 degraded variant per image in our Cell Painting dataset.

### Image quality assessment metric evaluation of degraded variant against reference
We then selected a collection of 5 full reference, image quality assessment metrics commonly used in microscopy and histology virtual staining work:
- Structural Similarity Index Measure (SSIM)
- Peak Signal-to-noise Ratio (PSNR)
- Deep Image Structure And Texture Similarity (DISTS)
- Learned Perceptual Image Patch Similarity (LPIPS)
- L1 (MAE)
And created foreground masked variants of SSIM and PSNR, inspired by the foreground aware training objective in Kalinin and colleagues (10.48550/arXiv.2507.05383):
- Foreground SSIM
- Foreground PSNR

Per every pair of degraded image variant and its original reference image, we evaluate our collection of 7 metrics, yielding 7 metric values per degraded varaint.

### Variance partitioning analysis
We then analyze with ANOVA variance partitioning and nested regression to what extents are variantions in metric values explained by the known degradation severity changes,
cell line indenitity of the images,
seeding densitiy label associated with the images,
and other known data inherent factors.

This allows us to quantify the absolute and relative metric sensitivity to degradation and susceptibility to confounding by data inherent variations.
