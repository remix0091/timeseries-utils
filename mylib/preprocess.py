import pandas as pd
import numpy as np
from mylib.outliers import remove_outliers, select_outlier_detection_method # подтягиваем функции для выбросов

def prepare_weekly_series(df, sid, thr=1e-5, agg="sum", verbose=True):
    """
    df: датафрейм с колонками week / series_id / value
    sid: какой ряд извлекаем
    thr: всё <= thr считаем пропуском (в т.ч. нули)
    agg: метод агрегации (sum, mean, last и т.п.)
    """

    # 1. Фильтрация нужного ряда
    sub = df[df["series_id"] == sid].copy()

    # 2. Преобразование недели в дату: типа "W25_23" в дату (понедельник недели)
    sub["week_dt"] = pd.to_datetime(
        sub["week"].str.extract(r"W(\d{1,2})_(\d{2})") # вытаскиваем номер недели и год (две цифры)
                    .apply(lambda x: f"20{x[1]}-W{x[0]}-1", axis=1),  # собираем дату формата ISO
        format="%G-W%V-%u"  # год-неделя-день недели (1 — понедельник)
    ) 

    # 3. Числовой тип и замена почти-нолей на NaN
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")  # ошибки как NaN
    mask_small = sub["value"].abs() <= thr   # создаём маску где значения маленькие
    n_small = mask_small.sum()  # считаем сколько таких
    sub.loc[mask_small, "value"] = np.nan  # и заменяем их на NaN

    # 4. Агрегация по неделям (с сохранением NaN если они были)
    grouped = (
        sub.groupby("week_dt", dropna=False)["value"]
        .agg(lambda x: x.iloc[0] if x.isna().all() else x.agg(agg))  # если все NaN — оставляем NaN 
        .to_frame()
    )

    # 5. Восстановление пропущенных недель
    full_idx = pd.date_range(grouped.index.min(), grouped.index.max(), freq="W-MON")  # все недели от и до
    merged = pd.DataFrame(index=full_idx).merge(grouped, left_index=True, right_index=True, how="left")  # подставляем пропуски

  
    if verbose:
        print(f"[ЛОГ] Ряд: {sid}")
        print(f"  - Даты: {grouped.index.min().date()} — {grouped.index.max().date()}")
        print(f"  - Преобразовано в NaN по порогу ({thr}): {n_small}")
        print(f"  - Пропущенных недель (NaN): {merged['value'].isna().sum()}")

    return merged["value"]   # возвращаем Series по value

def prepare_clean_series(df, sid, threshold=1e-5, agg="sum", verbose=True):
    log_meta = {}   # пишем лог
    log_meta["series_id"] = sid   # записываем номер ряда

    # Подготовка ряда
    series = prepare_weekly_series(df, sid, thr=threshold, agg=agg, verbose=False) # получаем ряд с пропусками по порогу
    n_missing = series.isna().sum()  # считаем сколько NaN
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

    return series_clean, log_meta, outlier_mask # возвращаем очищенный ряд, лог и маску выбросов


def drop_temporal_features(df, keywords=("lag", "shift", "relative", "time_since", "delta")):
    """
    Удаляет временные признаки из датафрейма на основе ключевых слов в названии колонок.

    df       : исходный датафрейм с фичами
    keywords : список ключевых слов для фильтрации (по умолчанию: временные сдвиги)

    Возвращает: датафрейм без временных признаков
    """
    drop_cols = [col for col in df.columns if any(k in col.lower() for k in keywords)]  # ищем все подходящие колонки
    if drop_cols:
        print(f"[ЛОГ] Удалены временные признаки: {drop_cols}")  # если нашли пишем какие
    else:
        print(f"[ЛОГ] Временные признаки не найдены.")
    return df.drop(columns=drop_cols, errors="ignore")  # удаляем


def create_features(series: pd.Series, lags=(1, 2, 3), window=3):
    """
    Строит простые признаки для обучения модели.
    Возвращает датафрейм с фичами и y.
    на вход ряд значений (Series), строим датафрейм признаков
    lags: на сколько шагов назад берём лаги
    window: скользящее среднее по окну

    """
    df = pd.DataFrame({"y": series})  # целевая переменная y
    for lag in lags:   # создаём лаговые признаки
        df[f"lag_{lag}"] = df["y"].shift(lag)  # сдвигаем y назад на lag
    df[f"rolling_mean_{window}"] = df["y"].rolling(window=window).mean()  # скользящее среднее

    # Дата признаки
    df["week"] = df.index.isocalendar().week
    df["month"] = df.index.month
    df["year"] = df.index.year

    return df.dropna()  # убираем строки где есть NaN (в начале из за лагов)
