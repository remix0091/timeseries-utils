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
def impute_series(series, enable=True, method="linear", verbose=True, **kw):
    """Заполняет NaN согласно указанному методу, если enable=True."""
    n_before = series.isna().sum()

    if not enable or n_before == 0:
        if verbose:
            print(f"[ЛОГ] Пропуски не заполнялись (enable={enable}, пропусков: {n_before})")
        return series

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