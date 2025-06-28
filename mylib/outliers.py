import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
import matplotlib.pyplot as plt  # 🔥 добавлен для визуализации

def select_outlier_detection_method(series):
    n = len(series.dropna())
    std = series.std()
    ac7 = series.autocorr(lag=7)
    method = "не определён"
    mask = None

    if n < 30:
        method = "Z-score (короткий ряд)"
        mask = np.abs(zscore(series, nan_policy="omit")) > 2.5
    elif ac7 > 0.4:
        method = "STL (сезонность)"
        resid = STL(series, period=7).fit().resid
        mask = np.abs(resid) > 2.5 * resid.std()
    elif std < 0.2:
        method = "IQR (низкий шум)"
        q1, q3 = series.quantile([.25, .75]); iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    elif std < 0.5:
        method = "LOF (средний шум)"
        try:
            lof = LocalOutlierFactor(n_neighbors=20)
            mask = lof.fit_predict(series.values.reshape(-1, 1)) == -1
        except ValueError:
            method = "Z-score (fallback после LOF)"
            mask = np.abs(zscore(series, nan_policy="omit")) > 2.5
    else:
        method = "PCA / IsolationForest"
        try:
            scaled = StandardScaler().fit_transform(series.values.reshape(-1, 1))
            comp = PCA(n_components=1).fit_transform(scaled)
            mask = np.abs(zscore(comp, nan_policy="omit")) > 2.5
        except ValueError:
            iso = IsolationForest(contamination=0.05, random_state=42)
            mask = iso.fit_predict(series.values.reshape(-1, 1)) == -1

    mask = pd.Series(np.asarray(mask).squeeze(), index=series.index)
    print(f"[ЛОГ] Метод обнаружения выбросов: {method}")
    print(f"[ЛОГ] Найдено выбросов: {mask.sum()}")
    return mask

def remove_outliers(series: pd.Series) -> pd.Series:
    mask = select_outlier_detection_method(series)
    out = series.copy()
    out[mask] = np.nan
    print(f"[ЛОГ] Выбросы удалены: {mask.sum()} точек заменено на NaN")
    return out

def plot_outliers(series, mask=None, title_prefix="Обнаруженные выбросы"):
    """Строит график временного ряда с выбросами"""
    if mask is None:
        mask = select_outlier_detection_method(series)

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series, label="Исходные данные")
    plt.scatter(series.index[mask], series[mask], color="red", label="Выбросы")
    plt.title(f"{title_prefix}")
    plt.legend()
    plt.grid(True)
    plt.show()
