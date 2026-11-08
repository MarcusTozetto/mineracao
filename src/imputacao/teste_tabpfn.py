"""Teste do TabPFN numa amostra de 10 mil partidas, contra MICE e mediana.

Os três métodos usam a mesma amostra, os mesmos 5 folds e as mesmas máscaras
(10% a 40% dos valores escondidos), então a comparação é direta. Este teste
não faz parte do pipeline (main.py); o resultado vai para
results/imputacao/teste_tabpfn.csv.

O TabPFN treina um modelo por variável (alvo = a variável, entradas = as
outras 13), usando o fold de treino como contexto, e prevê só as células
escondidas. Só um modelo fica na GPU por vez.

Configuração por variáveis de ambiente (padrão entre parênteses):
    TESTE_AMOSTRA       partidas da amostra (10000)
    TABPFN_ESTIMADORES  tamanho do ensemble do TabPFN (4)
    TABPFN_LOTE         linhas previstas por chamada; reduza se faltar memória (1000)
    TABPFN_PRECISAO     float32 ou auto (float32). Em placas AMD sem suporte
                        oficial (RX 6000), a precisão reduzida (auto) pode
                        falhar com "HIP error: invalid device function".

Placa AMD: antes de rodar, `newgrp render` e
`export HSA_OVERRIDE_GFX_VERSION=10.3.0`.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from tabpfn import TabPFNRegressor

from src.config import RESULTADOS_IMPUTACAO
from src.imputacao.protocolo import (
    COLUNAS, KFOLD, PERCENTUAIS, calcular_metricas, carregar_base, gerar_mascara,
)

AMOSTRA = int(os.environ.get("TESTE_AMOSTRA", 10_000))
N_ESTIMADORES = int(os.environ.get("TABPFN_ESTIMADORES", 4))
LOTE = int(os.environ.get("TABPFN_LOTE", 1_000))
PRECISAO = os.environ.get("TABPFN_PRECISAO", "float32")
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {DISPOSITIVO}"
      + (f" ({torch.cuda.get_device_name(0)})" if DISPOSITIVO == "cuda" else ""))
if DISPOSITIVO == "cpu":
    print("ATENÇÃO: sem GPU o TabPFN fica muito lento. Confira `newgrp render` "
          "e HSA_OVERRIDE_GFX_VERSION=10.3.0.")

X = carregar_base().sample(n=AMOSTRA, random_state=42).reset_index(drop=True)
print(f"Amostra: {X.shape} | estimadores: {N_ESTIMADORES} | lote: {LOTE} | precisão: {PRECISAO}")


def imputar_tabpfn(X_treino, ausentes):
    """Imputa cada máscara de `ausentes` (dict percentual -> matriz com NaN).

    Ajusta um modelo por variável e o usa em todos os percentuais antes de
    passar para a próxima, para manter só um modelo na GPU.
    """
    imputados = {p: a.copy() for p, a in ausentes.items()}
    tempo_fit = 0.0
    tempo_transform = {p: 0.0 for p in ausentes}

    for i, coluna in enumerate(COLUNAS):
        inicio = time.perf_counter()
        modelo = TabPFNRegressor(
            n_estimators=N_ESTIMADORES,
            device=DISPOSITIVO,
            inference_precision=torch.float32 if PRECISAO == "float32" else PRECISAO,
            random_state=42,
        )
        modelo.fit(np.delete(X_treino, i, axis=1), X_treino[:, i])
        tempo_fit += time.perf_counter() - inicio

        for p, a in ausentes.items():
            linhas = np.where(np.isnan(a[:, i]))[0]
            if len(linhas) == 0:
                continue
            inicio = time.perf_counter()
            entradas = np.delete(a[linhas], i, axis=1)
            previsoes = [modelo.predict(entradas[k:k + LOTE])
                         for k in range(0, len(entradas), LOTE)]
            imputados[p][linhas, i] = np.concatenate(previsoes)
            tempo_transform[p] += time.perf_counter() - inicio

        del modelo
        if DISPOSITIVO == "cuda":
            torch.cuda.empty_cache()
        print(f"    TabPFN {coluna:<4} ok")

    return imputados, tempo_fit, tempo_transform


resultados = []
inicio_geral = time.perf_counter()

for fold, (idx_treino, idx_teste) in enumerate(KFOLD.split(X), start=1):
    X_treino = X.iloc[idx_treino].to_numpy(dtype=float)
    X_teste = X.iloc[idx_teste].to_numpy(dtype=float)
    print(f"\nFold {fold}: treino {len(X_treino)} | teste {len(X_teste)}")

    mascaras = {p: gerar_mascara(fold, p, X_teste.shape) for p in PERCENTUAIS}
    ausentes = {p: np.where(m, np.nan, X_teste) for p, m in mascaras.items()}

    # MICE e mediana
    for nome, imputador in [
        ("MICE", IterativeImputer(random_state=17, max_iter=5, tol=1e-2,
                                  initial_strategy="mean", imputation_order="random")),
        ("Mediana", SimpleImputer(strategy="median")),
    ]:
        inicio = time.perf_counter()
        imputador.fit(X_treino)
        tempo_fit = time.perf_counter() - inicio
        for p in PERCENTUAIS:
            inicio = time.perf_counter()
            X_imp = imputador.transform(ausentes[p])
            tempo = time.perf_counter() - inicio
            resultados += calcular_metricas(nome, p, fold, X_teste, X_imp,
                                            mascaras[p], tempo_fit, tempo)

    # TabPFN
    imputados, tempo_fit, tempo_transform = imputar_tabpfn(X_treino, ausentes)
    for p in PERCENTUAIS:
        resultados += calcular_metricas("TabPFN", p, fold, X_teste, imputados[p],
                                        mascaras[p], tempo_fit, tempo_transform[p])

    gerais = pd.DataFrame(resultados).query("variavel == 'GERAL' and fold == @fold")
    print(gerais.pivot(index="percentual", columns="metodo", values="MAE").round(3))
    print(f"  tempo total até aqui: {(time.perf_counter() - inicio_geral) / 60:.1f} min")

    pd.DataFrame(resultados).to_csv(RESULTADOS_IMPUTACAO / "teste_tabpfn.csv", index=False)

df = pd.DataFrame(resultados)
gerais = df[df["variavel"] == "GERAL"]

print("\n===== MAE MÉDIO (5 folds) =====")
print(gerais.pivot_table(index="percentual", columns="metodo", values="MAE",
                         aggfunc="mean").round(3))
print("\n===== DESVIO DO MAE =====")
print(gerais.pivot_table(index="percentual", columns="metodo", values="MAE",
                         aggfunc="std").round(3))
print("\n===== TEMPO TOTAL MÉDIO POR FOLD (s) =====")
print(gerais.pivot_table(index="percentual", columns="metodo", values="tempo_total",
                         aggfunc="mean").round(1))

df.to_csv(RESULTADOS_IMPUTACAO / "teste_tabpfn.csv", index=False)
print("\nResultados salvos em results/imputacao/teste_tabpfn.csv")
