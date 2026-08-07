import os
import sys
from pathlib import Path

import pandas as pd


def _running_in_ipykernel() -> bool:
    """Return whether the current process is running under an IPython kernel."""
    return "ipykernel" in sys.modules


def require_env(name: str, default: str | None = None) -> str:
    """Return a required configuration value.

    Under ipykernel, environment lookup is skipped and ``default`` is required.
    Otherwise, the environment variable is used, falling back to ``default``
    when provided.
    """
    if _running_in_ipykernel():
        if default is None:
            raise RuntimeError(
                f"A default value for {name!r} is required when running under ipykernel."
            )
        value = default
    else:
        value = os.environ.get(name, default)

    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable {name!r} is not set or is empty.")

    return value.strip()


def require_positive_int_env(
    name: str,
    default: int | None = None,
) -> int:
    """Return a required configuration value parsed as a positive integer."""
    raw_value = require_env(
        name,
        default=None if default is None else str(default),
    )

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name!r} must be an integer, but received {raw_value!r}."
        ) from exc

    if value <= 0:
        raise RuntimeError(f"Environment variable {name!r} must be positive, but received {value}.")

    return value


def require_bool_env(name: str, default: bool | None = None) -> bool:
    """Return a required configuration value parsed as a boolean."""
    raw_value = require_env(
        name,
        default=None if default is None else str(default),
    )

    if raw_value.lower() in ("true", "1", "yes"):
        return True
    elif raw_value.lower() in ("false", "0", "no"):
        return False
    else:
        raise RuntimeError(
            f"Environment variable {name!r} must be a boolean (true/false), but received {raw_value!r}."
        )


def _merge_path_filename(
    row: pd.Series,
    chan: str,
    path_col_template: str = "PathName_{}",
    file_col_template: str = "FileName_{}",
) -> Path:
    return Path(row[path_col_template.format(chan)]) / row[file_col_template.format(chan)]


def build_dataset_inputs(
    loaddata: pd.DataFrame,
    input_chan: str,
    target_chan: str,
    profile: pd.DataFrame | None = None,
    unique_id_cols: list[str] | None = None,
    obj_coord_x_col: str | None = "Metadata_Cells_Location_Center_X",
    obj_coord_y_col: str | None = "Metadata_Cells_Location_Center_Y",
    **kwargs,
) -> tuple[pd.DataFrame, list[dict] | None]:

    if unique_id_cols is None:
        unique_id_cols = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]

    df_meta = loaddata.loc[:, [col for col in loaddata.columns if col.startswith("Metadata_")]]
    if "Metadata_time_point" not in df_meta.columns and "time_point" in loaddata.columns:
        df_meta["Metadata_time_point"] = loaddata["time_point"]

    input_paths = loaddata.apply(lambda row: _merge_path_filename(row, input_chan), axis=1).values
    target_paths = loaddata.apply(lambda row: _merge_path_filename(row, target_chan), axis=1).values

    image_file_index = pd.DataFrame(
        {
            input_chan: input_paths,
            target_chan: target_paths,
        }
    )

    hcat_df = pd.concat([image_file_index, df_meta], axis=1)
    if profile is not None:
        df1 = hcat_df.reset_index(drop=True).copy()
        df2 = profile.reset_index(drop=True).copy()
        df1["_iloc1"] = df1.index
        df2["_iloc2"] = df2.index

        merge_df = pd.merge(
            df1,
            df2,
            on=unique_id_cols + ["Metadata_time_point"],
            how="inner",
            validate="one_to_many",
        )

        mapping = merge_df.groupby("_iloc1")["_iloc2"].apply(list).to_dict()
    else:
        merge_df = hcat_df

    if (
        obj_coord_x_col is None
        or obj_coord_y_col is None
        or obj_coord_x_col not in merge_df.columns
        or obj_coord_y_col not in merge_df.columns
    ):
        pt_mapping = None
    else:
        image_file_index_clean = df1.iloc[list(mapping.keys()), :]
        pt_mapping = [
            {
                "X": merge_df.loc[merge_df["_iloc1"] == idx][obj_coord_x_col].values,
                "Y": merge_df.loc[merge_df["_iloc1"] == idx][obj_coord_y_col].values,
            }
            for idx in image_file_index_clean.index
        ]

    return image_file_index_clean, pt_mapping
