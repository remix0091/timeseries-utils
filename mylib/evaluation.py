import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mylib.preprocess import prepare_weekly_series
from mylib.outliers import remove_outliers
from mylib.imputation import impute_series

# ⬇️ MAPE — средняя абсолютная процентная ошибка
def mape(y_true, y_pred):
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.nan

# ⬇️ WAPE — взвешенная абсолютная процентная ошибка
def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_pred - y_true)) / denom * 100

# ⬇️ Оценка качества восстановления разных методов
def evaluate_methods_with_mask(df, series_id, step=5):
    """
    df        : DataFrame с колонками week / series_id / value
    series_id : ID ряда (например, "#1")
    step      : через сколько точек маскировать значения

    Возвращает таблицу с метриками MSE, MAE, MAPE, WAPE
    """
    full_series = remove_outliers(prepare_weekly_series(df, series_id))

    # искусственно маскируем каждую step-ую точку
    masked = full_series.copy()
    valid_idx = full_series.dropna().iloc[::step].index
    masked.loc[valid_idx] = np.nan
    na_idx = masked[masked.isna()].index

    methods = ["linear", "mean", "median", "ffill", "bfill", "spline", "rolling"]
    results = []

    for method in methods:
        filled = impute_series(masked.copy(), method=method)

        y_true = full_series.loc[na_idx].values
        y_pred = filled.loc[na_idx].values

        # удаляем пары с NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        if len(y_true) < 3:
            results.append([method, np.nan, np.nan, np.nan, np.nan])
            continue

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape_ = mape(y_true, y_pred)
        wape_ = wape(y_true, y_pred)

        results.append([method, mse, mae, mape_, wape_])

    return pd.DataFrame(results, columns=["method", "MSE", "MAE", "MAPE_%", "WAPE_%"])
