"""Helper utilities for dataset creation from CellProfiler loaddata and profiling outputs"""

import pathlib

import pandas as pd

UNIQUE_ID_COLS = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]

def _normalize_key_columns(
    df: pd.DataFrame, 
    cols: list[str]
) -> pd.DataFrame:
    """
    Normalize the key columns of a dataframe for consistent comparison.

    :params df: Input dataframe containing metadata columns.
    :params cols: List of column names to normalize.
    :returns: A dataframe with normalized key columns.
    """
    out = df.loc[:, cols].copy()
    if "Metadata_Site" in cols:
        out["Metadata_Site"] = pd.to_numeric(out["Metadata_Site"], errors="coerce").astype("Int64")
    for c in cols:
        if c != "Metadata_Site":
            out[c] = out[c].astype("string")
    return out


def _filter_by_profile_intersection(
    loaddata: pd.DataFrame,
    profile: pd.DataFrame,
    unique_id_cols: list[str] | None = None, 
    normalize: bool = False
) -> pd.DataFrame:
    """
    Filter the loaddata dataframe to only retain rows that have corresponding 
        entries in the profile dataframe based on unique identifier columns.

    :params loaddata: Input loaddata dataframe containing metadata columns.
    :params profile: Profile dataframe containing metadata columns.
    :params unique_id_cols: List of column names used as unique identifiers for merging.
    :params normalize: Whether to normalize the key columns before comparison.
    :returns: A filtered loaddata dataframe containing only rows with matching profile entries.
    """
    if unique_id_cols is None:
        unique_id_cols = UNIQUE_ID_COLS

    keys = _normalize_key_columns(loaddata, unique_id_cols) if normalize else loaddata.loc[:, unique_id_cols]
    profile_keys = (
        _normalize_key_columns(profile, unique_id_cols) if normalize else profile.loc[:, unique_id_cols]
        .dropna(subset=unique_id_cols)
        .drop_duplicates()
    )
    keep_mask = (
        keys.merge(profile_keys, on=unique_id_cols, how="left", indicator=True)["_merge"]
        .eq("both")
        .to_numpy()
    )

    return loaddata.loc[keep_mask].reset_index(drop=True)


def _merge_path_filename(
    row: pd.Series, 
    chan: str,
    path_col_template: str = "PathName_{}",
    file_col_template: str = "FileName_{}"
) -> pathlib.Path:
    """Small helper for merging path and filename columns into a pathlib.Path object."""
    return pathlib.Path(row[path_col_template.format(chan)]) / row[file_col_template.format(chan)]


def build_dataset_inputs(
    loaddata: pd.DataFrame,
    channels: list[str],
    profile: pd.DataFrame | None = None,
    unique_id_cols: list[str] | None = None, 
    obj_coord_x_col: str | None = "Metadata_Cells_Location_Center_X",
    obj_coord_y_col: str | None = "Metadata_Cells_Location_Center_Y",
    **kwargs
) -> tuple[pd.DataFrame, list[dict] | None]:

    if unique_id_cols is None:
        unique_id_cols = UNIQUE_ID_COLS
    
    loaddata = _filter_by_profile_intersection(
        loaddata, profile, unique_id_cols=unique_id_cols, **kwargs
    ) if profile is not None else loaddata
    
    df_meta = loaddata.loc[:, [col for col in loaddata.columns if col.startswith("Metadata_")]]
    
    if "Metadata_time_point" not in df_meta.columns and "time_point" in loaddata.columns:
        df_meta['Metadata_time_point'] = loaddata['time_point']

    channel_paths = {
        channel: loaddata.apply(
            _merge_path_filename,
            axis=1,
            chan=channel,
        ).values
        for channel in channels
    }

    image_file_index = pd.DataFrame(
        channel_paths
    )

    hcat_df = pd.concat([image_file_index, df_meta], axis=1)
    if profile is not None:
        df1 = hcat_df.reset_index(drop=True).copy()
        df2 = profile.reset_index(drop=True).copy()
        df1['_iloc1'] = df1.index
        df2['_iloc2'] = df2.index

        merge_df = pd.merge(
            df1,
            df2,
            on=unique_id_cols,
            how="inner",
            validate="one_to_many"
        )

        mapping = (
            merge_df.groupby('_iloc1')['_iloc2']
            .apply(list)
            .to_dict()
        )
    else:
        merge_df = hcat_df
        

    if obj_coord_x_col is None or obj_coord_y_col is None or obj_coord_x_col not in merge_df.columns or obj_coord_y_col not in merge_df.columns:
        pt_mapping = None    
    else:
        image_file_index_clean = df1.iloc[
            list(mapping.keys()),:
        ]
        pt_mapping = [{
                'X': merge_df.loc[merge_df['_iloc1'] == idx][obj_coord_x_col].values,
                'Y': merge_df.loc[merge_df['_iloc1'] == idx][obj_coord_y_col].values
            }
            for idx in image_file_index_clean.index]
    
    return image_file_index_clean, pt_mapping
