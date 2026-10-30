"""Miniproyecto 1: la serie del grupo, de las líneas base al mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y la tabla en resultados/metricas.csv. Las
partes marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import holidays  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, ingenuo)

warnings.filterwarnings("ignore")
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ============================================================
# 1. La serie
# ============================================================
# Cambie estos cuatro datos por los de la serie del grupo y documente la
# fuente en datos/README.md. La serie de ejemplo solo sirve para probar que
# el guion corre.
ARCHIVO = "datos/IPN32221N_cajas_carton_corrugado.csv"
FECHA = "observation_date"
VALOR = "IPN32221N"
PERIODO = 12

# Lo que se hizo con los huecos y con los datos atípicos, en una línea. Va
# a la columna `decision` de la tabla de métricas.
DECISION = "Sin huecos; atipicos winsorizados al +-3*IQR del residuo STL (COVID 2020, rebote 2022)"


def preparar(serie):
    """Aplica la decision del grupo sobre huecos y datos atipicos.

    La serie IPN32221N no tiene meses vacios (0 NaN), asi que no hay huecos
    que rellenar. Los valores atipicos no son errores de medicion sino dos
    episodios economicos reales -el desplome de la COVID (abr-jun 2020) y el
    rebote post-pandemia (feb-jun 2022)-, que inflan la varianza del residuo
    y pueden dominar el ajuste del SARIMA. La decision es *winsorizarlos*:
    recortar al limite, sin borrar meses, solo los puntos cuyo residuo de una
    STL robusta (periodo 12) se sale de la banda Q1-3*IQR .. Q3+3*IQR. Asi se
    conserva la forma y el numero de observaciones de la serie.

    Devuelve una serie sin NaN.
    """
    serie = serie.astype(float)
    if serie.isna().any():  # por robustez: si hubiera huecos, se interpolan
        serie = serie.interpolate(method="time")

    stl = STL(serie, period=PERIODO, robust=True).fit()
    resid = stl.resid
    q1, q3 = resid.quantile(0.25), resid.quantile(0.75)
    iqr = q3 - q1
    bajo, alto = q1 - 3 * iqr, q3 + 3 * iqr

    resid_rec = resid.clip(lower=bajo, upper=alto)
    exceso = resid - resid_rec            # 0 donde no hay atipico
    limpia = serie - exceso               # recorta solo los puntos fuera de banda
    limpia.name = serie.name

    n_rec = int((exceso != 0).sum())
    print(f"Preparar: 0 huecos; {n_rec} atipicos winsorizados "
          f"(banda residuo STL [{bajo:.2f}, {alto:.2f}])")
    return limpia


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.set_title(f"{VALOR}: serie mensual")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. Descomposición
# ============================================================

def descomponer(serie):
    """STL robusta con periodo 12; guarda figuras/stl.png e imprime la lectura."""
    stl = STL(serie, period=PERIODO, robust=True).fit()

    figura = stl.plot()
    figura.set_size_inches(9, 7)
    figura.suptitle(f"{VALOR}: descomposicion STL robusta (periodo {PERIODO})")
    figura.tight_layout()
    figura.savefig(FIGURAS / "stl.png", dpi=120)
    plt.close(figura)

    tendencia = stl.trend
    estacional = stl.seasonal
    resid = stl.resid

    # Perfil estacional: valor tipico de cada mes (promedio del componente).
    perfil = estacional.groupby(estacional.index.month).mean().sort_values()
    mes_bajo, mes_alto = perfil.index[0], perfil.index[-1]
    r_grande = resid.abs().idxmax()

    print("STL - tendencia: inicia en {:.1f} ({:%Y-%m}), maximo {:.1f} "
          "({:%Y-%m}), termina en {:.1f} ({:%Y-%m})".format(
              tendencia.iloc[0], tendencia.index[0], tendencia.max(),
              tendencia.idxmax(), tendencia.iloc[-1], tendencia.index[-1]))
    print("STL - estacional: mes mas alto = {} ({:+.2f}), mes mas bajo = {} "
          "({:+.2f})".format(mes_alto, perfil.iloc[-1], mes_bajo, perfil.iloc[0]))
    print("STL - residuo mas grande: {:%Y-%m} ({:+.2f}); desv. tipica resid = "
          "{:.2f}".format(r_grande, resid[r_grande], resid.std()))
    return stl


# ============================================================
# 3. Estacionariedad e identificación
# ============================================================

def _adf(x, reg):
    """Valor p de la prueba ADF (H0: raiz unitaria, es decir NO estacionaria)."""
    stat, p, *_ = adfuller(x.dropna(), regression=reg, autolag="AIC")
    return stat, p


def estacionariedad(serie):
    """ADF de las cuatro versiones, elige d y D, y guarda la ACF y la PACF.

    regression: en el nivel se usa 'c' (la serie tiene media distinta de cero
    pero su tendencia sube y baja, no es una recta determinista); en las
    diferencias tambien 'c' (permite deriva). Como control se reporta 'ct'
    sobre el nivel. H0 de ADF es raiz unitaria; p<0.05 => estacionaria.
    """
    pruebas = [
        ("serie (nivel)",              serie,                   "c"),
        ("serie (nivel, tendencia)",   serie,                   "ct"),
        ("dif regular d=1",            serie.diff(1),           "c"),
        ("dif estacional D=1 (lag12)", serie.diff(PERIODO),     "c"),
        ("ambas d=1 y D=1",            serie.diff(PERIODO).diff(1), "c"),
    ]
    print("ADF (H0: raiz unitaria; p<0.05 => estacionaria):")
    for nombre, x, reg in pruebas:
        stat, p = _adf(x, reg)
        print(f"  {nombre:28s} reg={reg:2s}  p={p:.4f}  stat={stat:+.3f}")

    # Eleccion: estacionalidad fuerte y gran caida de varianza con la
    # diferencia estacional -> D=1; tras D=1 queda raiz unitaria regular
    # (p alto), que desaparece al anadir la diferencia regular -> d=1.
    d, D = 1, 1
    print(f"  => se eligen d={d}, D={D}")

    # ACF y PACF de la serie estacionaria (d=1, D=1).
    estac = serie.diff(PERIODO).diff(1).dropna()
    figura, ejes = plt.subplots(2, 1, figsize=(9, 6))
    plot_acf(estac, lags=36, ax=ejes[0], zero=False)
    plot_pacf(estac, lags=36, ax=ejes[1], method="ywm", zero=False)
    ejes[0].set_title(f"ACF de la serie con d={d}, D={D}")
    ejes[1].set_title(f"PACF de la serie con d={d}, D={D}")
    figura.tight_layout()
    figura.savefig(FIGURAS / "acf_pacf.png", dpi=120)
    plt.close(figura)

    # Rezagos fuera de las bandas +-1.96/sqrt(n) (para la lectura del informe).
    from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf
    n = len(estac)
    banda = 1.96 / np.sqrt(n)
    va = _acf(estac, nlags=36, fft=False)
    vp = _pacf(estac, nlags=36, method="ywm")
    sig_acf = [k for k in range(1, 37) if abs(va[k]) > banda]
    sig_pacf = [k for k in range(1, 37) if abs(vp[k]) > banda]
    print(f"  ACF  fuera de banda (+-{banda:.3f}): {sig_acf}")
    print(f"  PACF fuera de banda (+-{banda:.3f}): {sig_pacf}")
    return d, D


# ============================================================
# 4. Modelos clásicos
# ============================================================
# Cada modelo es una función modelo(entrenamiento, h) que devuelve h
# pronósticos y solo usa `entrenamiento`. Así entra al backtest común.

def _ajustar(entrenamiento, orden, estacional, exog=None):
    """Ajusta un SARIMA(X). enforce_* en False para que la rejilla converja."""
    return SARIMAX(entrenamiento, order=orden, seasonal_order=estacional,
                   exog=exog, enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)


def elegir_sarima(serie, d, D):
    """Rejilla pequena (p,q)(P,Q) por AIC, con d y D fijos; diagnostica residuos.

    La rejilla la sugiere la ACF/PACF: parte regular de orden bajo (MA) y una
    estacional de MA que necesita orden 2 por los satelites alrededor del
    lag 12. Se comparan por AIC; del elegido se revisa Ljung-Box (sobre los
    residuos estandarizados, descartando el burn-in de la diferenciacion) y se
    guarda figuras/residuos.png. Devuelve (orden, orden_estacional).
    """
    ordenes = [(0, 1), (1, 1), (0, 2), (1, 2)]   # (p, q)
    estacionales = [(0, 1), (1, 1), (0, 2), (1, 2)]  # (P, Q)
    filas = []
    for p, q in ordenes:
        for P, Q in estacionales:
            try:
                aic = _ajustar(serie, (p, d, q), (P, D, Q, PERIODO)).aic
            except Exception:
                aic = np.inf
            filas.append({"orden": (p, d, q), "estacional": (P, D, Q, PERIODO),
                          "AIC": aic})
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    print(f"Rejilla SARIMA por AIC ({len(tabla)} modelos, d={d}, D={D}):")
    for _, r in tabla.head(5).iterrows():
        print(f"  {r['orden']}{r['estacional']}  AIC={r['AIC']:.2f}")
    orden = tabla.iloc[0]["orden"]
    estacional = tabla.iloc[0]["estacional"]
    print(f"  => elegido {orden}{estacional}  AIC={tabla.iloc[0]['AIC']:.2f}")

    # Diagnostico de residuos del elegido (ajustado con toda la serie).
    ajuste = _ajustar(serie, orden, estacional)
    burn = d + D * PERIODO
    resid = pd.Series(ajuste.standardized_forecasts_error[0],
                      index=serie.index).iloc[burn:]
    k = orden[0] + orden[2] + estacional[0] + estacional[2]  # p+q+P+Q
    lb = acorr_ljungbox(resid, lags=[12, 24], model_df=k, return_df=True)
    print("  Ljung-Box residuos:  LB(12) p={:.3f}   LB(24) p={:.3f}".format(
        lb["lb_pvalue"].iloc[0], lb["lb_pvalue"].iloc[1]))

    figura = ajuste.plot_diagnostics(figsize=(9, 7), lags=36)
    figura.suptitle(f"Residuos de SARIMA{orden}{estacional}")
    figura.tight_layout()
    figura.savefig(FIGURAS / "residuos.png", dpi=120)
    plt.close(figura)
    return orden, estacional


def modelo_sarima(orden, orden_estacional):
    """Funcion modelo(entrenamiento, h) del SARIMA elegido para el backtest."""
    def modelo(entrenamiento, h):
        ajuste = _ajustar(entrenamiento, orden, orden_estacional)
        return ajuste.forecast(h)
    return modelo


def exogenas(serie):
    """Dias habiles de cada mes (EE. UU.): variable de calendario del indice.

    Se conoce de antemano para cualquier mes futuro -depende solo del
    calendario, no de la serie-, asi que nunca introduce fuga. Cubre el indice
    de la serie y los 12 meses siguientes, para poder pronosticar.
    """
    idx = pd.date_range(serie.index[0],
                        serie.index[-1] + pd.offsets.MonthBegin(12), freq="MS")
    festivos = holidays.US()
    conteo = []
    for mes in idx:
        dias = pd.date_range(mes.replace(day=1),
                             mes + pd.offsets.MonthEnd(0), freq="D")
        conteo.append(sum(1 for dia in dias
                          if dia.weekday() < 5 and dia.date() not in festivos))
    return pd.DataFrame({"dias_habiles": conteo}, index=idx, dtype=float)


def modelo_sarimax(orden, orden_estacional, x):
    """Funcion modelo(entrenamiento, h) de un SARIMAX con la exogena x.

    Usa x.loc[entrenamiento.index] para ajustar y x en las h fechas
    siguientes para pronosticar; esas fechas son calendario, no datos de la
    serie posteriores al origen, de modo que no hay fuga.
    """
    def modelo(entrenamiento, h):
        x_tr = x.loc[entrenamiento.index]
        x_fut = x.loc[fechas_siguientes(entrenamiento, h)]
        ajuste = _ajustar(entrenamiento, orden, orden_estacional, exog=x_tr)
        return ajuste.forecast(h, exog=x_fut)
    return modelo


# ============================================================
# 5. La tabla de métricas
# ============================================================

def fila(nombre, serie, modelo):
    _, promedio = backtest(serie, modelo)
    return {"modelo": nombre, "decision": DECISION,
            "RMSE": round(promedio["RMSE"], 2), "MAE": round(promedio["MAE"], 2),
            "MAPE": round(promedio["MAPE"], 2)}


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    filas = [fila("ingenuo", serie, ingenuo),
             fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]

    try:
        descomponer(serie)
    except NotImplementedError as error:
        pendiente("parte 2", error)

    try:
        d, D = estacionariedad(serie)
    except NotImplementedError as error:
        pendiente("parte 3", error)
        d = D = None

    if d is not None:
        try:
            orden, estacional = elegir_sarima(serie, d, D)
            filas.append(fila(f"SARIMA{orden}{estacional}", serie,
                              modelo_sarima(orden, estacional)))
            x = exogenas(serie)
            filas.append(fila(f"SARIMAX{orden}{estacional} + exógenas", serie,
                              modelo_sarimax(orden, estacional, x)))
        except NotImplementedError as error:
            pendiente("parte 4", error)

    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.drop(columns="decision").to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])


if __name__ == "__main__":
    main()
