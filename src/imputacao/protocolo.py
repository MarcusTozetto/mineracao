"""Protocolo comum da comparação de imputação.

Reúne o que precisa ser idêntico entre os métodos para que a comparação seja
justa: a base, os folds, as máscaras de ausência e as métricas. Os scripts dos
métodos novos (mediana, TabPFN) usam este módulo; os resultados ficam no mesmo
formato dos experimentos de MICE, KNN, MissForest e GAIN.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.config import DADOS_PROCESSADOS, RESULTADOS_IMPUTACAO

COLUNAS = [
    "HTHG", "HTAG",
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",
]

PERCENTUAIS = [0.10, 0.20, 0.30, 0.40]

KFOLD = KFold(n_splits=5, shuffle=True, random_state=42)


def carregar_base():
    """Partidas com as 14 estatísticas completas (mesmo recorte dos outros métodos)."""
    df = pd.read_csv(DADOS_PROCESSADOS / "futebol_limpo.csv", low_memory=False)
    return df[COLUNAS].dropna().reset_index(drop=True)


def gerar_mascara(fold, percentual, formato):
    """Máscara MCAR do conjunto de teste.

    A semente segue a mesma fórmula dos outros experimentos, então o fold 1 com
    10% de ausência esconde exatamente as mesmas células em todos os métodos.
    """
    semente = 42 + fold * 100 + int(percentual * 100)
    rng = np.random.default_rng(semente)
    return rng.random(formato) < percentual


def calcular_metricas(metodo, percentual, fold, X_teste, X_imputado, mascara,
                      tempo_fit, tempo_transform):
    """Linhas de resultado: uma geral e uma por variável."""
    reais = np.asarray(X_teste)
    imputados = np.asarray(X_imputado)

    mse = mean_squared_error(reais[mascara], imputados[mascara])
    linhas = [{
        "metodo": metodo,
        "percentual": percentual * 100,
        "fold": fold,
        "variavel": "GERAL",
        "MAE": mean_absolute_error(reais[mascara], imputados[mascara]),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": np.nan,
        "tempo_fit": tempo_fit,
        "tempo_transform": tempo_transform,
        "tempo_total": tempo_fit + tempo_transform,
    }]

    for i, coluna in enumerate(COLUNAS):
        m = mascara[:, i]
        real, imp = reais[m, i], imputados[m, i]
        mse_col = mean_squared_error(real, imp)
        linhas.append({
            "metodo": metodo,
            "percentual": percentual * 100,
            "fold": fold,
            "variavel": coluna,
            "MAE": mean_absolute_error(real, imp),
            "MSE": mse_col,
            "RMSE": np.sqrt(mse_col),
            "R2": r2_score(real, imp),
            "tempo_fit": np.nan,
            "tempo_transform": np.nan,
            "tempo_total": np.nan,
        })
    return linhas


def salvar_resultados(resultados, nome):
    """Salva resultados completos, resumo geral e R² por variável."""
    df = pd.DataFrame(resultados)
    df.to_csv(RESULTADOS_IMPUTACAO / f"resultados_{nome}_kfold.csv", index=False)

    gerais = df[df["variavel"] == "GERAL"]
    resumo = gerais.groupby("percentual").agg(
        MAE_medio=("MAE", "mean"), MAE_desvio=("MAE", "std"),
        MSE_medio=("MSE", "mean"), MSE_desvio=("MSE", "std"),
        RMSE_medio=("RMSE", "mean"), RMSE_desvio=("RMSE", "std"),
        tempo_fit_medio=("tempo_fit", "mean"),
        tempo_transform_medio=("tempo_transform", "mean"),
        tempo_total_medio=("tempo_total", "mean"),
    )
    resumo.to_csv(RESULTADOS_IMPUTACAO / f"resumo_{nome}_kfold.csv")

    por_variavel = df[df["variavel"] != "GERAL"]
    resumo_r2 = por_variavel.groupby(["percentual", "variavel"]).agg(
        R2_medio=("R2", "mean"), R2_desvio=("R2", "std"),
    )
    resumo_r2.to_csv(RESULTADOS_IMPUTACAO / f"resumo_r2_{nome}_kfold.csv")
    return resumo
