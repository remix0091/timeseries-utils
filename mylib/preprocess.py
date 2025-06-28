import pandas as pd
import numpy as np
from mylib.outliers import remove_outliers, select_outlier_detection_method

def prepare_weekly_series(df, sid, thr=1e-5, agg="sum", verbose=True):
    """Из датафрейма df берём ряд sid, приводим недели к датам,
       убираем дубликаты, вставляем пропущенные недели, мелкие числа → NaN."""
    
    sub = df[df["series_id"] == sid].copy()

    sub["week_dt"] = pd.to_datetime(
        sub["week"].str.extract(r"W(\d{1,2})_(\d{2})")
                    .apply(lambda x: f"20{x[1]}-W{x[0]}-1", axis=1),
        format="%G-W%V-%u")

    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    n_small = (sub["value"] < thr).sum()
    sub.loc[sub["value"] < thr, "value"] = np.nan

    grouped = sub.groupby("week_dt")["value"].agg(agg).to_frame()

    full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="W-MON")
    full = pd.DataFrame(index=full_idx)
    merged = full.merge(grouped, left_index=True, right_index=True, how="left")

    if verbose:
        print(f"[ЛОГ] Ряд: {sid}")
        print(f"  - Даты: {grouped.index.min().date()} — {grouped.index.max().date()}")
        print(f"  - Преобразовано в NaN по порогу ({thr}): {n_small}")
        print(f"  - Пропущенных недель (NaN): {merged['value'].isna().sum()}")

    return merged["value"]

def prepare_clean_series(df, sid, threshold=1e-5, agg="sum", verbose=True):
    log_meta = {}
    log_meta["series_id"] = sid

    # Подготовка ряда
    series = prepare_weekly_series(df, sid, thr=threshold, agg=agg, verbose=False)
    n_missing = series.isna().sum()
    log_meta["missing_count"] = n_missing
    log_meta["missing_indices"] = series[series.isna()].index.tolist()
    if verbose:
        print(f"[ЛОГ] Пропусков до удаления выбросов: {n_missing}")

    # Находим выбросы один раз
    outlier_mask = select_outlier_detection_method(series)
    n_outliers = outlier_mask.sum()
    log_meta["outlier_count"] = n_outliers
    log_meta["outlier_indices"] = outlier_mask[outlier_mask].index.tolist()
    if verbose:
        print(f"[ЛОГ] Выбросов удалено: {n_outliers}")

    # Удаляем выбросы
    series_clean = series.copy()
    series_clean[outlier_mask] = np.nan

    return series_clean, log_meta, outlier_mask
