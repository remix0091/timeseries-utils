import pandas as pd
import numpy as np
from mylib.preprocess import prepare_weekly_series
from mylib.outliers import remove_outliers, plot_outliers, select_outlier_detection_method

def fill_rolling_mean(s: pd.Series, window: int = 5) -> pd.Series:
    """
    Заполняем NaN средним из `window` последних (!) уже-заполненных значений.
    Алгоритм идёт слева-направо, поэтому свежее восстановленное
    значение сразу попадает в окно для следующих точек.
    """
    filled = s.copy()
    for i in range(len(filled)):
        if pd.isna(filled.iat[i]):
            start = max(0, i - window)           # левая граница окна
            mean_val = filled.iloc[start:i].mean()
            filled.iat[i] = mean_val
    return filled


def find_nan_blocks(series):
    """Возвращает список (индексы начала, индексы конца, длина) всех подряд идущих блоков NaN"""
    blocks = []
    in_block = False
    start = None

    for i, is_na in enumerate(series.isna()):
        if is_na and not in_block:
            in_block = True
            start = i
        elif not is_na and in_block:
            in_block = False
            end = i
            blocks.append((start, end - 1, end - start))

    if in_block:  # если NaN до конца
        blocks.append((start, len(series) - 1, len(series) - start))

    return blocks
applied_blocks = []


def fill_with_rules(series):
    series = series.copy()
    blocks = find_nan_blocks(series)

    for start, end, length in blocks:
        left = max(0, start - 10)
        right = min(len(series), end + 10 + 1)
        frag = series.iloc[left:right].copy()

        if length <= 2:
            method_used = "linear"
            applied_blocks.append((start, end, method_used))

            filled = frag.interpolate("linear", limit_direction="both")

        elif length <= 10:
            method_used = "ffill+bfill+spline"
            applied_blocks.append((start, end, method_used))

            filled = frag.interpolate("linear", limit_direction="both").interpolate("spline",
                                                      order=3,
                                                      limit_area="inside")
        else:
            method_used = "rolling-mean"
            applied_blocks.append((start, end, method_used))
            for i in range(start, end + 1):
                if i < len(series):
                    window = series.iloc[max(0, i-10):i]
                    series.iloc[i] = window.mean()
            print(f"    блок {start}:{end}  len={length:2d}  →  {method_used}")
            continue   # длинный блок уже записан в series

        # вставляем обратно по индексам
        block_idx = series.index[start:end + 1]
        series.loc[block_idx] = filled.reindex(block_idx).values
        print(f"    блок {start}:{end}  len={length:2d}  →  {method_used}")

    return series

def impute_series(series, enable=True, method="linear", verbose=True, **kw):
    """Заполняет NaN согласно указанному методу, если enable=True."""
    n_before = series.isna().sum()

    if not enable or n_before == 0:
        if verbose:
            print(f"[ЛОГ] Пропуски не заполнялись (enable={enable}, пропусков: {n_before})")
        return series
    
    if method == "auto":
        if verbose:
            print("[ЛОГ] Автоматический режим: применяем fill_with_rules(...)")
        return fill_with_rules(series)

    if method == "mean":
        filled = series.fillna(series.mean())
    elif method == "median":
        filled = series.fillna(series.median())
    elif method == "linear":
        filled = series.interpolate("linear", limit_direction="both", **kw)
    elif method == "spline":
        filled = series.interpolate("spline", order=kw.get("order", 2), limit_direction="both")
    elif method == "ffill":
        filled = series.ffill()
    elif method == "bfill":
        filled = series.bfill()
    elif method == "constant":
        filled = series.fillna(kw.get("value", 0))
    elif method == "rolling":
        win = kw.get("window", 5)
        filled = fill_rolling_mean(series, window=win)
    else:
        raise ValueError("неизвестный метод")

    n_after = filled.isna().sum()

    if verbose:
        print(f"[ЛОГ] Метод заполнения: {method}")
        print(f"  - Пропусков до: {n_before}")
        print(f"  - Пропусков после: {n_after}")
        if kw:
            print(f"  - Параметры: {kw}")

    return filled

def process_all_series(df, config, verbose=True, plot=True):
    """
    df      : исходный DataFrame с week / series_id / value
    config  : словарь настроек по каждому ряду
    verbose : выводить логи
    plot    : строить график выбросов
    return  : словарь {series_id: {"raw", "clean", "filled", "log"}}
    """
    output = {}

    for sid, params in config.items():
        log_meta = {"series_id": sid}

        raw = prepare_weekly_series(df, sid)
        log_meta["missing_count"] = raw.isna().sum()
        log_meta["missing_indices"] = raw[raw.isna()].index.tolist()

        outlier_mask = select_outlier_detection_method(raw)
        clean = raw.copy()
        clean[outlier_mask] = np.nan

        log_meta["outlier_count"] = outlier_mask.sum()
        log_meta["outlier_indices"] = outlier_mask[outlier_mask].index.tolist()

        filled = impute_series(
            clean,
            enable=params.get("enable", True),
            method=params.get("method", "linear"),
            **{k: v for k, v in params.items() if k not in {"enable", "method"}}
        )

        log_meta["final_missing"] = filled.isna().sum()

        output[sid] = {
            "raw": raw,
            "clean": clean,
            "filled": filled,
            "log": log_meta
        }

        if verbose:
            print(f"[ЛОГ] Ряд {sid}: пропусков={log_meta['missing_count']}, выбросов={log_meta['outlier_count']}, итоговых NaN={log_meta['final_missing']}")
        if plot:
            plot_outliers(raw, mask=outlier_mask, title_prefix=f"Выбросы в ряде {sid}")
        if verbose or plot:
            print("-" * 40)

    return output