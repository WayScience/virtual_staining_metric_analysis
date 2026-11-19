"""
Public entry points for TIFF/outline ingestion and single-plane reads.
Dataset-specific selection, processingrecipes belong to the caller, not this storage backend.
"""

from utils.fov_ingestion.ingest import ingest, load_plane, outlines_to_labels, plan

__all__ = ["ingest", "load_plane", "outlines_to_labels", "plan"]
