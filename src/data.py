# data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Tuple, List


def _dense(arr) -> np.ndarray:
    """Return a dense numpy array whether 'arr' is sparse or already dense."""
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def _build_single_feature_windows(
    df_long: pd.DataFrame,
    id_col: str,
    year_col: str,
    feature_col: str,
    sequence_length: int,
    encoder: OneHotEncoder,
):
    """
    Build sliding-window X,y for a single feature (independent), using a shared OHE for id.

    Returns
    -------
    X : np.ndarray, shape (n, sequence_length + num_ids)
    y : np.ndarray, shape (n,)
    meta : pd.DataFrame with columns [id_col, 'target_year'] aligned with rows in X,y
    """
    df_long = df_long.copy()
    # fillna per id
    df_long[feature_col] = df_long.groupby(id_col)[feature_col].transform(
        lambda g: g.fillna(g.mean())
    )

    # pivot: rows=id, cols=year, vals=feature
    wide = df_long.pivot(index=id_col, columns=year_col, values=feature_col)
    wide = wide.reindex(sorted(wide.columns), axis=1)

    # must have at least seq_len + 1 non-NA (k history + 1 target)
    wide = wide.dropna(axis=0, thresh=sequence_length + 1)
    if wide.empty:
        num_ids = len(encoder.categories_[0])
        empty_X = np.zeros((0, sequence_length + num_ids))
        empty_y = np.zeros((0,))
        empty_meta = pd.DataFrame(columns=[id_col, "target_year"])
        return empty_X, empty_y, empty_meta

    years = list(wide.columns)
    ids = list(wide.index)

    X_list, y_list, meta_rows = [], [], []
    for _id in ids:
        series = wide.loc[_id].values
        oh = _dense(encoder.transform([[str(_id)]])).ravel()
        for i in range(sequence_length, len(years)):
            window = series[i - sequence_length : i]
            target = series[i]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X_list.append(np.concatenate([window, oh]))
            y_list.append(target)
            meta_rows.append({id_col: str(_id), "target_year": int(years[i])})

    X = np.asarray(X_list, float)
    y = np.asarray(y_list, float)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def load_and_prepare_multi(
    filepath: str,
    feature_cols: List[str],
    sequence_length: int = 3,
    id_col: str = "id",
    year_col: str = "year",
):
    """
    Build supervised datasets for ALL requested features, independently.

    Returns
    -------
    X_map    : dict feature -> X (n_f, sequence_length + num_ids)
    y_map    : dict feature -> y (n_f,)
    meta_map : dict feature -> pd.DataFrame with columns [id_col, 'target_year']
    encoder  : OneHotEncoder fitted on all ids in the file (reuse in predict)
    """
    df = pd.read_excel(filepath)

    # basic checks
    need = {id_col, year_col, *feature_cols}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {sorted(missing)}")

    # ensure dtypes
    df = df[[id_col, year_col] + feature_cols].copy()
    df[year_col] = df[year_col].astype(int)
    df[id_col] = df[id_col].astype(str)

    # global id list for encoder (all ids that appear)
    all_ids = sorted(df[id_col].unique().tolist())

    # sklearn compatibility: new uses 'sparse_output', old uses 'sparse'
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(np.array(all_ids).reshape(-1, 1))

    # one long df to reuse
    df_long = df.sort_values([id_col, year_col]).reset_index(drop=True)

    X_map, y_map, meta_map = {}, {}, {}
    for feat in feature_cols:
        Xf, yf, metaf = _build_single_feature_windows(
            df_long=df_long,
            id_col=id_col,
            year_col=year_col,
            feature_col=feat,
            sequence_length=sequence_length,
            encoder=encoder,
        )
        X_map[feat] = Xf
        y_map[feat] = yf
        meta_map[feat] = metaf

    return X_map, y_map, meta_map, encoder
