import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor  # метод для поиска выбросов по соседям
from sklearn.preprocessing import StandardScaler  # нормализация данных
from sklearn.ensemble import IsolationForest  # ещё один способ для детекта выбросов
from sklearn.decomposition import PCA  # понижение размерности
from statsmodels.tsa.seasonal import STL   # разложение временных рядов (тренд, сезон, остаток)
from scipy.stats import zscore  # стандартное отклонение в z-оценке
import matplotlib.pyplot as plt  # добавлен для визуализации

def select_outlier_detection_method(series):
    n = len(series.dropna())  # считаем сколько непустых значений в ряду
    std = series.std()   # стандартное отклонение, насколько ряд шумный
    ac7 = series.autocorr(lag=7)   # автокорреляция с лагом 7 (есть ли сезонность недельная)
    method = "не определён"   # сюда потом запишем название выбранного метода
    mask = None   # итоговая маска выбросов (True если выброс)

    if n < 30:
        method = "Z-score (короткий ряд)"
        mask = np.abs(zscore(series, nan_policy="omit")) > 2.5  # если z > 2.5 то это выброс

    elif ac7 > 0.4:  # если сильная сезонность — используем STL
        method = "STL (сезонность)"
        resid = STL(series, period=7).fit().resid  # берём остаток после вычитания тренда и сезона
        mask = np.abs(resid) > 2.5 * resid.std()   # если остаток слишком большой то выброс
    elif std < 0.2: # очень ровный ряд, пробуем IQR
        method = "IQR (низкий шум)"
        q1, q3 = series.quantile([.25, .75]); iqr = q3 - q1# квартильный размах
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr) # если вышли за границы то выброс
    elif std < 0.5:# средне-ровный, можно попробовать LOF
        method = "LOF (средний шум)"
        try:
            lof = LocalOutlierFactor(n_neighbors=20) # строим локальную модель
            mask = lof.fit_predict(series.values.reshape(-1, 1)) == -1  # -1 = выброс
        except ValueError:
            method = "Z-score (fallback после LOF)"  # если LOF не сработал — откат на z-score
            mask = np.abs(zscore(series, nan_policy="omit")) > 2.5
    else:  # если ряд шумный, используем что то посложнее, PCA или IsolationForest
        method = "PCA / IsolationForest"
        try:
            scaled = StandardScaler().fit_transform(series.values.reshape(-1, 1))  # нормируем
            comp = PCA(n_components=1).fit_transform(scaled)  # понижаем размерность
            mask = np.abs(zscore(comp, nan_policy="omit")) > 2.5  # ищем выбросы уже в проекции
        except ValueError:
            iso = IsolationForest(contamination=0.05, random_state=42)  # fallback метод
            mask = iso.fit_predict(series.values.reshape(-1, 1)) == -1  # -1 это выброс
    # приводим маску к pandas.Series и делаем, чтобы у неё был тот же индекс что и у серии
    mask = pd.Series(np.asarray(mask).squeeze(), index=series.index)
    print(f"[ЛОГ] Метод обнаружения выбросов: {method}")
    print(f"[ЛОГ] Найдено выбросов: {mask.sum()}")
    return mask # возвращаем маску выбросов

def remove_outliers(series: pd.Series) -> pd.Series:
    mask = select_outlier_detection_method(series)  # сначала выбираем метод и получаем маску
    out = series.copy()
    out[mask] = np.nan # заменяем выбросы на NaN
    print(f"[ЛОГ] Выбросы удалены: {mask.sum()} точек заменено на NaN")
    return out   # возвращаем очищенный ряд

def plot_outliers(series, mask=None, title_prefix="Обнаруженные выбросы"):
    """Строит график временного ряда с выбросами"""
    if mask is None:
        mask = select_outlier_detection_method(series)  # если маска не передана, то сами ищем

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series, label="Исходные данные")
    plt.scatter(series.index[mask], series[mask], color="red", label="Выбросы")
    plt.title(f"{title_prefix}")
    plt.legend()
    plt.grid(True)
    plt.show()
