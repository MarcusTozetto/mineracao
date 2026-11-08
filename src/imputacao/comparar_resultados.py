import pandas as pd
import numpy as np


from src.config import RESULTADOS_IMPUTACAO


# ============================================================
# 1. CARREGAR RESULTADOS
# ============================================================

nomes = ["mice", "missforest", "gain", "knn", "tabpfn", "mediana"]

arquivos = [
    RESULTADOS_IMPUTACAO / f"resultados_{nome}_kfold.csv"
    for nome in nomes
]


dfs = []

for arquivo in arquivos:
    # métodos que ainda não foram rodados ficam de fora
    if not arquivo.exists():
        print("Sem resultados (ignorado):", arquivo.name)
        continue
    df = pd.read_csv(arquivo)
    dfs.append(df)


resultados = pd.concat(
    dfs,
    ignore_index=True
)


print("Métodos encontrados:")
print(resultados["metodo"].unique())

print("\nQuantidade de registros:")
print(len(resultados))


# ============================================================
# 2. SEPARAR RESULTADOS GERAIS
# ============================================================

gerais = resultados[
    resultados["variavel"] == "GERAL"
].copy()


print("\nResultados gerais:")
print(
    gerais[
        [
            "metodo",
            "percentual",
            "fold",
            "MAE",
            "MSE",
            "RMSE"
        ]
    ]
)


# ============================================================
# 3. RESUMO GERAL
# ============================================================

resumo = (

    gerais

    .groupby(
        [
            "metodo",
            "percentual"
        ]
    )

    .agg(

        MAE_medio=(
            "MAE",
            "mean"
        ),

        MAE_desvio=(
            "MAE",
            "std"
        ),

        MSE_medio=(
            "MSE",
            "mean"
        ),

        MSE_desvio=(
            "MSE",
            "std"
        ),

        RMSE_medio=(
            "RMSE",
            "mean"
        ),

        RMSE_desvio=(
            "RMSE",
            "std"
        ),

        tempo_fit_medio=(
            "tempo_fit",
            "mean"
        ),

        tempo_transform_medio=(
            "tempo_transform",
            "mean"
        ),

        tempo_total_medio=(
            "tempo_total",
            "mean"
        )
    )

    .reset_index()
)


print(
    "\n===== RESUMO DOS MÉTODOS ====="
)

print(
    resumo.to_string(
        index=False
    )
)


resumo.to_csv(
    RESULTADOS_IMPUTACAO / "comparacao_metodos_kfold.csv",
    index=False
)


# ============================================================
# 4. TABELA APENAS COM MAE
# ============================================================

tabela_mae = (

    resumo

    .pivot(
        index="percentual",
        columns="metodo",
        values="MAE_medio"
    )
)


print(
    "\n===== MAE MÉDIO ====="
)

print(tabela_mae)


tabela_mae.to_csv(
    RESULTADOS_IMPUTACAO / "comparacao_mae_kfold.csv"
)


# ============================================================
# 5. TABELA DE TEMPO
# ============================================================

tabela_tempo = (

    resumo

    .pivot(
        index="percentual",
        columns="metodo",
        values="tempo_total_medio"
    )
)


print(
    "\n===== TEMPO MÉDIO ====="
)

print(tabela_tempo)


tabela_tempo.to_csv(
    RESULTADOS_IMPUTACAO / "comparacao_tempo_kfold.csv"
)


# ============================================================
# 6. R² POR VARIÁVEL
# ============================================================

por_variavel = resultados[
    resultados["variavel"] != "GERAL"
].copy()


resumo_r2 = (

    por_variavel

    .groupby(
        [
            "metodo",
            "percentual",
            "variavel"
        ]
    )

    .agg(

        R2_medio=(
            "R2",
            "mean"
        ),

        R2_desvio=(
            "R2",
            "std"
        )
    )

    .reset_index()
)


resumo_r2.to_csv(
    RESULTADOS_IMPUTACAO / "comparacao_r2_kfold.csv",
    index=False
)


print(
    "\n===== R² POR VARIÁVEL ====="
)

print(
    resumo_r2.to_string(
        index=False
    )
)

# Os testes estatísticos (Friedman e Wilcoxon) ficam em
# src/imputacao/testes_estatisticos.py
