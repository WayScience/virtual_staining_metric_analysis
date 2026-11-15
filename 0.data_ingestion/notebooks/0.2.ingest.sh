fov-ingest /home/weishanli/Waylab/virtual_staining_metric_analysis/metadata/loaddata_for_degradation.csv \
  --outline-root /home/weishanli/mnt/bandicoot/PCCMA_data/PedCP90_profiles/sqlite_outputs/Round_1_data \
  --dataset-id pedcp90-reference \
  --output /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_ref \
  --skip-missing-outlines \
  --workers 8 --pixel-size-um 0.593
