import pandas as pd
import numpy as np
from mylib.outliers import remove_outliers, select_outlier_detection_method

def prepare_weekly_series(df, sid, thr=1e-5, agg="sum", verbose=True):
    """
    df: датафрейм с колонками week / series_id / value
    sid: какой ряд извлекаем
    thr: всё <= thr считаем пропуском (в т.ч. нули)
    agg: метод агрегации (sum, mean, last и т.п.)
    """

    # 1. Фильтрация нужного ряда
    sub = df[df["series_id"] == sid].copy()

    # 2. Преобразование недели в дату
    sub["week_dt"] = pd.to_datetime(
        sub["week"].str.extract(r"W(\d{1,2})_(\d{2})")
                    .apply(lambda x: f"20{x[1]}-W{x[0]}-1", axis=1),
        format="%G-W%V-%u"
    )

    # 3. Числовой тип и замена почти-нолей на NaN
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    mask_small = sub["value"].abs() <= thr
    n_small = mask_small.sum()
    sub.loc[mask_small, "value"] = np.nan

    # 4. Агрегация по неделям (с сохранением NaN если они были)
    grouped = (
        sub.groupby("week_dt", dropna=False)["value"]
        .agg(lambda x: x.iloc[0] if x.isna().all() else x.agg(agg))
        .to_frame()
    )

    # 5. Восстановление пропущенных недель
    full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="W-MON")
    merged = pd.DataFrame(index=full_idx).merge(grouped, left_index=True, right_index=True, how="left")

  
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


def drop_temporal_features(df, keywords=("lag", "shift", "relative", "time_since", "delta")):
    """
    Удаляет временные признаки из датафрейма на основе ключевых слов в названии колонок.

    df       : исходный датафрейм с фичами
    keywords : список ключевых слов для фильтрации (по умолчанию: временные сдвиги)

    Возвращает: датафрейм без временных признаков
    """
    drop_cols = [col for col in df.columns if any(k in col.lower() for k in keywords)]
    if drop_cols:
        print(f"[ЛОГ] Удалены временные признаки: {drop_cols}")
    else:
        print(f"[ЛОГ] Временные признаки не найдены.")
    return df.drop(columns=drop_cols, errors="ignore")


def create_features(series: pd.Series, lags=(1, 2, 3), window=3):
    """
    Строит простые признаки для обучения модели.
    Возвращает датафрейм с фичами и y.
    """
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df[f"rolling_mean_{window}"] = df["y"].rolling(window=window).mean()

    # Дата признаки
    df["week"] = df.index.isocalendar().week
    df["month"] = df.index.month
    df["year"] = df.index.year

    return df.dropna()
