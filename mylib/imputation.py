import pandas as pd
import numpy as np
from mylib.preprocess import prepare_weekly_series
from mylib.outliers import remove_outliers, plot_outliers, select_outlier_detection_method

#скользящее среднее
def fill_rolling_mean(s: pd.Series, window: int = 5) -> pd.Series: # функция принимает ряд и окно (по умолч 5)
    """
    заполняем NaN средним из window последних  уже заполненных значений
    Алгоритм идёт слева-направо, поэтому свежее восстановленное значение сразу попадает в окно для следующих точек
    """
    filled = s.copy() # делаем копию ряда, чтобы оригинал не трогать
    for i in range(len(filled)): # идём по всем элементам
        if pd.isna(filled.iat[i]): # если текущий элемент — NaN
            start = max(0, i - window)           # левая граница окна, считаем индекс начала окна, чтобы не выйти за границы
            mean_val = filled.iloc[start:i].mean() # берём среднее значение в этом окне
            filled.iat[i] = mean_val  # подставляем вместо NaN
    return filled # возвращаем уже заполненный ряд


def find_nan_blocks(series):
    """Возвращает список (индексы начала, индексы конца, длина) всех подряд идущих блоков NaN"""
    blocks = []  # сюда будем записывать найденные блоки
    in_block = False # флаг, внутри блока или нет
    start = None # где начинается блок

    for i, is_na in enumerate(series.isna()): # перебираем значения и проверяем, NaN ли они
        if is_na and not in_block: # если наткнулись на NaN и ещё не в блоке
            in_block = True # ставим флаг
            start = i  # запоминаем начало блока
        elif not is_na and in_block: # если блок закончился (встретили не-NaN)
            in_block = False
            end = i
            blocks.append((start, end - 1, end - start)) # сохраняем блок (начало, конец, длина)

    if in_block:  # если NaN до конца
        blocks.append((start, len(series) - 1, len(series) - start))

    return blocks # возвращаем все найденные NaN-блоки

applied_blocks = []  # глобальный список, в который запишем что и чем заполняли

def fill_with_rules(series):
    series = series.copy()
    blocks = find_nan_blocks(series)   # ищем блоки NaN в ряду

    for start, end, length in blocks:  # перебираем каждый блок
        left = max(0, start - 10)  # берём левую границу с запасом
        right = min(len(series), end + 10 + 1)  # и правую тоже с запасом
        frag = series.iloc[left:right].copy()  # вырезаем кусок ряда вокруг пропуска

        if length <= 2:  # если короткий пропуск
            method_used = "linear"  # используем линейную интерполяцию
            applied_blocks.append((start, end, method_used)) # записываем в лог

            filled = frag.interpolate("linear", limit_direction="both")   # интерполируем

        elif length <= 10:  # если пропуск побольше
            method_used = "ffill+bfill+spline"   # комбинируем методы
            applied_blocks.append((start, end, method_used))

            filled = frag.interpolate("linear", limit_direction="both").interpolate("spline",
                                                      order=3,
                                                      limit_area="inside") # интерполяция + сплайн внутри данных
        else:  # длинный пропуск
            method_used = "rolling-mean"  # заполняем средним по окну
            applied_blocks.append((start, end, method_used))
            for i in range(start, end + 1):
                if i < len(series):
                    window = series.iloc[max(0, i-10):i]  # берём окно до точки
                    series.iloc[i] = window.mean()   # и подставляем среднее
            print(f"    блок {start}:{end}  len={length:2d}  →  {method_used}")
            continue   # длинный блок уже записан в series

        # вставляем обратно по индексам
        block_idx = series.index[start:end + 1]  # получаем индексы блока
        series.loc[block_idx] = filled.reindex(block_idx).values  # вставляем обратно в основной ряд
        print(f"    блок {start}:{end}  len={length:2d}  →  {method_used}")

    return series  # возвращаем результат

def impute_series(series, enable=True, method="linear", verbose=True, **kw):
    """Заполняет NaN согласно указанному методу, если enable=True."""
    n_before = series.isna().sum()  # считаем сколько было NaN

    if not enable or n_before == 0:   # если выключено или пропусков нет — выходим
        if verbose:
            print(f"[ЛОГ] Пропуски не заполнялись (enable={enable}, пропусков: {n_before})")
        return series
    
    if method == "auto":  # если авто-режим
        if verbose:
            print("[ЛОГ] Автоматический режим: применяем fill_with_rules(...)")
        return fill_with_rules(series) 
  # разные варианты чем заполнять
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
        raise ValueError("неизвестный метод")  # если метод не поддерживается

    n_after = filled.isna().sum()  # сколько NaN осталось
 
    if verbose:  # если включен лог, печатаем инфу
        print(f"[ЛОГ] Метод заполнения: {method}")
        print(f"  - Пропусков до: {n_before}")
        print(f"  - Пропусков после: {n_after}")
        if kw:
            print(f"  - Параметры: {kw}")

    return filled  # возвращаем заполненный ряд

def process_all_series(df, config, verbose=True, plot=True):
    """
    df      : исходный DataFrame с week / series_id / value
    config  : словарь настроек по каждому ряду
    verbose : выводить логи
    plot    : строить график выбросов
    return  : словарь {series_id: {"raw", "clean", "filled", "log"}}
    """
    output = {}   # сюда складываем результат

    for sid, params in config.items():   # идём по каждому ряду
        log_meta = {"series_id": sid} # лог по этому ряду
 
        raw = prepare_weekly_series(df, sid)   # вытаскиваем нужный временной ряд по id
        log_meta["missing_count"] = raw.isna().sum()
        log_meta["missing_indices"] = raw[raw.isna()].index.tolist()

        outlier_mask = select_outlier_detection_method(raw)  # определяем где выбросы
        clean = raw.copy()
        clean[outlier_mask] = np.nan # заменяем выбросы на NaN

        log_meta["outlier_count"] = outlier_mask.sum()
        log_meta["outlier_indices"] = outlier_mask[outlier_mask].index.tolist()

        filled = impute_series(
            clean,
            enable=params.get("enable", True),
            method=params.get("method", "linear"),
            **{k: v for k, v in params.items() if k not in {"enable", "method"}} # передаём доп параметры
        )

        log_meta["final_missing"] = filled.isna().sum() # сколько NaN осталось

        output[sid] = {  # собираем всё в словарь
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

    return output  # возвращаем результат по всем рядам