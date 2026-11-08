"""Linha de base: imputação pela mediana de cada variável no treino.

Serve de referência mínima: um método só vale a pena se for melhor que isto.
"""
import time

from sklearn.impute import SimpleImputer

from src.imputacao.protocolo import (
    KFOLD, PERCENTUAIS, calcular_metricas, carregar_base, gerar_mascara,
    salvar_resultados,
)

X = carregar_base()
print("Base completa:", X.shape)

resultados = []

for fold, (idx_treino, idx_teste) in enumerate(KFOLD.split(X), start=1):
    X_treino, X_teste = X.iloc[idx_treino], X.iloc[idx_teste]

    inicio = time.perf_counter()
    imputador = SimpleImputer(strategy="median").fit(X_treino)
    tempo_fit = time.perf_counter() - inicio

    for percentual in PERCENTUAIS:
        mascara = gerar_mascara(fold, percentual, X_teste.shape)

        inicio = time.perf_counter()
        X_imputado = imputador.transform(X_teste.mask(mascara))
        tempo_transform = time.perf_counter() - inicio

        linhas = calcular_metricas("Mediana", percentual, fold, X_teste, X_imputado,
                                   mascara, tempo_fit, tempo_transform)
        resultados += linhas
        print(f"Mediana | fold {fold} | {percentual:.0%} | MAE {linhas[0]['MAE']:.4f}")

resumo = salvar_resultados(resultados, "mediana")
print("\n===== RESUMO MEDIANA + 5-FOLD =====")
print(resumo)
