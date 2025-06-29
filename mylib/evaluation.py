import numpy as np  # библиотека для числовых операций (работа с массивами, формулы...)
import pandas as pd # работа с таблицами (DataFrame, Series)
from sklearn.metrics import mean_squared_error, mean_absolute_error # стандартные метрики ошибок из sklearn
from mylib.preprocess import prepare_weekly_series # функция для подготовки временного ряда к анализу
from mylib.outliers import remove_outliers # удаляет аномалии из ряда
from mylib.imputation import impute_series # функция для заполнения пропусков в ряде
import matplotlib.pyplot as plt # для построения графиков

# MAPE — средняя абсолютная процентная ошибка
def mape(y_true, y_pred):
     
    # Вычисляет среднюю абсолютную процентную ошибку между правильными значениями и предсказаниями.
    # Используется когда важна относительная точность прогноза.
    # y_true — правильные значения,эталон
    # y_pred — предсказанные значения, результат восстановления
    
    mask = y_true != 0 # чтобы не делить на 0
    if np.any(mask): # если остались значения не равные 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.nan # если не с чем сравнивать вернём NaN

# WAPE — взвешенная абсолютная процентная ошибка, лучше работает с выбросавми и нестабильными рядами
def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true)) # сумма всех фактических значений по модулю
    if denom == 0:
        return np.nan # если делить не на что  вернём NaN
    return np.sum(np.abs(y_pred - y_true)) / denom * 100 # сумма ошибок / сумма значений

# Durbin–Watson
def durbin_watson_safe(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    безопасный расчёт Durbin–Watson исключая пропуски
    показывает есть ли автокорреляция ошибок (остатков).
    Чем ближе к 2 тем лучше - нет зависимости между ошибками
    """
    residuals = y_true - y_pred # считаем остатки = факт - прогноз
    residuals = pd.Series(residuals).dropna() # преобразуем в Series и удаляем пропуски

    if len(residuals) < 3: # если осталось слишком мало точек, не считаем
        return np.nan

    diff = np.diff(residuals)# разности между соседними остатками
    return np.sum(diff**2) / np.sum(residuals**2)# формула Durbin–Watson

# Оценка качества восстановления разных методов
def evaluate_methods_with_mask(df, series_id, step=5, plot=True):
    """
    df        : DataFrame с колонками week / series_id / value
    series_id : ID ряда (например, "#1")
    step      : через сколько точек маскировать значения
    plot      : рисовать график или нет

    Возвращает таблицу с метриками MSE, MAE, MAPE, WAPE
    """
    full_series = remove_outliers(prepare_weekly_series(df, series_id)) # готовим и очищаем ряд

    # искусственно маскируем каждую step-ую точку
    masked = full_series.copy() # копия ряда, в который будем вставлять пропуски
    valid_idx = full_series.dropna().iloc[::step].index # берём каждую stepую ненулевую точку
    masked.loc[valid_idx] = np.nan # заменяем её на NaN
    na_idx = masked[masked.isna()].index # сохраняем список индексов где nan

    methods = ["linear", "mean", "median", "ffill", "bfill", "spline", "rolling"]
    results = [] # сюда запишем результаты метрик
    predictions = {} # сюда запишем восстановленные значения для графика

    for method in methods:
        filled = impute_series(masked.copy(), method=method) # восстанавливаем пропуски выбранным методом

        y_true = full_series.loc[na_idx].values# истинные значения что было до маски
        y_pred = filled.loc[na_idx].values# значения, восстановленные методом

        # удаляем пары с NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred)) # фильтруем пары, где есть NaN
        y_true, y_pred = y_true[mask], y_pred[mask]# оставляем только валидные

        predictions[method] = (na_idx[mask], y_pred)  # сохраняем предсказания

        if len(y_true) < 3: # если сравнивать нечего — пропускаем
            results.append([method, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
# считаем метрики качества
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape_ = mape(y_true, y_pred)
        wape_ = wape(y_true, y_pred)
        dw = durbin_watson_safe(y_true, y_pred)
        results.append([method, mse, mae, mape_, wape_, dw])# добавляем строку в таблицу


    if plot:# если включён режим графиков
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(full_series, label="Оригинальный ряд", color="black", linewidth=1.5)
        plt.scatter(na_idx, full_series.loc[na_idx], label="Замаскировано", color="gray", marker="x")

        for method, (x, y) in predictions.items():
            plt.scatter(x, y, label=method, alpha=0.6)

        plt.title(f"Сравнение восстановления (ряд {series_id})")
        plt.xlabel("Дата")
        plt.ylabel("Значение")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results, columns=["method", "MSE", "MAE", "MAPE_%", "WAPE_%", "DW"])# таблица метрик



def evaluate_methods_on_custom_mask(series: pd.Series, mask_idx, plot=True, methods=None, method_params=None):
    """
    Сравнивает методы восстановления на пользовательских пропусках

    series   : исходный pd.Series
    mask_idx : список индексов, окон [(start, end)], или их сочетание
    plot     : отображать график
    methods — список методов восстановления (если None — используем все)
    method_params — словарь с параметрами для методов, пример {"spline": {"order": 3}}

    Возвращает: DataFrame с метриками
    """

    full_series = series.copy() # копия исходного ряда

    # аподдержка смешанного списка: [(start, end), index1, index2, ...]
    final_idx = [] # сюда соберём все индексы NaN

    for item in mask_idx:
        if isinstance(item, tuple) and len(item) == 2:
            # это окно (start, end)
            start, end = item
            final_idx.extend(series.index[start:end + 1]) # добавляем от start до end
        elif pd.Timestamp(item) in series.index:
            # одиночный индекс
            final_idx.append(item)
        else:
            continue

    mask_idx = pd.Index(final_idx)  # окончательная маска

    masked = full_series.copy()
    masked.loc[mask_idx] = np.nan # вставляем NaN по маске

    if methods is None:
        methods = ["linear", "mean", "median", "ffill", "bfill", "spline", "rolling"]
    if method_params is None:
        method_params = {}
    results = []
    predictions = {}

    for method in methods:
        params = method_params.get(method, {}) # получаем параметры метода
        filled = impute_series(masked.copy(), method=method, **params) # восстанавливаем

        y_true = full_series.loc[mask_idx].values # правильные значения 
        y_pred = filled.loc[mask_idx].values # предсказанные

        ok = ~(np.isnan(y_true) | np.isnan(y_pred)) # отсекаем NaN
        y_true, y_pred = y_true[ok], y_pred[ok]
        predictions[method] = (mask_idx[ok], y_pred)

        if len(y_true) < 3:
            results.append([method, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape_ = mape(y_true, y_pred)
        wape_ = wape(y_true, y_pred)
        dw = durbin_watson_safe(y_true, y_pred)
        results.append([method, mse, mae, mape_, wape_, dw])


    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(series, label="Оригинальный ряд", color="black", linewidth=1.5)
        plt.scatter(mask_idx, full_series.loc[mask_idx], label="Маскировано", color="gray", marker="x")

        for method, (x, y) in predictions.items():
            plt.scatter(x, y, label=method, alpha=0.6)

        plt.title("Сравнение методов восстановления (ручная маска)")
        plt.xlabel("Дата")
        plt.ylabel("Значение")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results, columns=["method", "MSE", "MAE", "MAPE_%", "WAPE_%", "DW"])


