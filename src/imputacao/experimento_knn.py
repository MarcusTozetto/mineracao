import pandas as pd
import numpy as np
import time

from src.imputacao.protocolo import KFOLD, N_FOLDS
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from src.config import DADOS_PROCESSADOS, RESULTADOS_IMPUTACAO


# ============================================================
# 1. CARREGAR BASE
# ============================================================

df = pd.read_csv(
    DADOS_PROCESSADOS / "futebol_limpo.csv",
    low_memory=False
)


colunas_numericas = [
    "HTHG", "HTAG",
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR"
]


# Mantemos apenas casos completos porque precisamos conhecer
# o valor verdadeiro para avaliar posteriormente a imputação.
X = (
    df[colunas_numericas]
    .dropna()
    .reset_index(drop=True)
    .copy()
)


print(
    "Subconjunto de casos completos:",
    X.shape
)


# ============================================================
# 2. CONFIGURAÇÕES
# ============================================================

percentuais = [
    0.10,
    0.20,
    0.30,
    0.40
]


kfold = KFOLD  # mesma divisão para todos os métodos (src/imputacao/protocolo.py)


resultados = []


# ============================================================
# 3. K-FOLD
# ============================================================

for fold, (
    indices_treino,
    indices_teste
) in enumerate(
    kfold.split(X),
    start=1
):


    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)


    X_treino = (
        X.iloc[
            indices_treino
        ]
        .copy()
    )


    X_teste = (
        X.iloc[
            indices_teste
        ]
        .copy()
    )


    print(
        "Treino:",
        X_treino.shape
    )

    print(
        "Teste:",
        X_teste.shape
    )


    # ========================================================
    # 4. CRIAR KNN
    # ========================================================

    imputador = KNNImputer(
        n_neighbors=10,
        weights="uniform"
    )


    # ========================================================
    # 5. FIT
    # ========================================================

    inicio_fit = (
        time.perf_counter()
    )


    imputador.fit(
        X_treino
    )


    fim_fit = (
        time.perf_counter()
    )


    tempo_fit = (
        fim_fit
        - inicio_fit
    )


    print(
        f"Tempo de treinamento: "
        f"{tempo_fit:.4f} segundos"
    )


    # ========================================================
    # 6. TESTAR CADA PERCENTUAL
    # ========================================================

    for percentual in percentuais:


        print(
            f"\nKNN | "
            f"Fold {fold} | "
            f"{percentual * 100:.0f}%"
        )


        # ====================================================
        # MESMA MÁSCARA DOS OUTROS MÉTODOS
        # ====================================================

        seed_mascara = (
            42
            + fold * 100
            + int(
                percentual * 100
            )
        )


        rng = np.random.default_rng(
            seed_mascara
        )


        mascara = (
            rng.random(
                X_teste.shape
            )
            < percentual
        )


        X_teste_missing = (
            X_teste.mask(
                mascara
            )
        )


        # ====================================================
        # 7. IMPUTAÇÃO
        # ====================================================

        inicio_transform = (
            time.perf_counter()
        )


        X_imputado = (
            imputador.transform(
                X_teste_missing
            )
        )


        fim_transform = (
            time.perf_counter()
        )


        tempo_transform = (
            fim_transform
            - inicio_transform
        )


        tempo_total = (
            tempo_fit
            + tempo_transform
        )


        # ====================================================
        # 8. MÉTRICAS GERAIS
        # ====================================================

        valores_reais = (
            X.to_numpy()[indices_teste][
                mascara
            ]
        )


        valores_imputados = (
            X_imputado[
                mascara
            ]
        )


        mae = mean_absolute_error(
            valores_reais,
            valores_imputados
        )


        mse = mean_squared_error(
            valores_reais,
            valores_imputados
        )


        rmse = np.sqrt(
            mse
        )


        print(
            "MAE:",
            mae
        )

        print(
            "MSE:",
            mse
        )

        print(
            "RMSE:",
            rmse
        )

        print(
            "Tempo transform:",
            tempo_transform,
            "segundos"
        )

        print(
            "Tempo total:",
            tempo_total,
            "segundos"
        )


        resultados.append({

            "metodo":
                "KNN",

            "percentual":
                percentual * 100,

            "fold":
                fold,

            "variavel":
                "GERAL",

            "MAE":
                mae,

            "MSE":
                mse,

            "RMSE":
                rmse,

            "R2":
                np.nan,

            "tempo_fit":
                tempo_fit,

            "tempo_transform":
                tempo_transform,

            "tempo_total":
                tempo_total
        })


        # ====================================================
        # 9. MÉTRICAS POR VARIÁVEL
        # ====================================================

        for i, coluna in enumerate(
            colunas_numericas
        ):


            mascara_coluna = (
                mascara[:, i]
            )


            real = (
                X_teste
                .iloc[:, i]
                .to_numpy()[
                    mascara_coluna
                ]
            )


            imputado = (
                X_imputado[
                    :,
                    i
                ][
                    mascara_coluna
                ]
            )


            mae_coluna = (
                mean_absolute_error(
                    real,
                    imputado
                )
            )


            mse_coluna = (
                mean_squared_error(
                    real,
                    imputado
                )
            )


            rmse_coluna = (
                np.sqrt(
                    mse_coluna
                )
            )


            r2_coluna = (
                r2_score(
                    real,
                    imputado
                )
            )


            resultados.append({

                "metodo":
                    "KNN",

                "percentual":
                    percentual * 100,

                "fold":
                    fold,

                "variavel":
                    coluna,

                "MAE":
                    mae_coluna,

                "MSE":
                    mse_coluna,

                "RMSE":
                    rmse_coluna,

                "R2":
                    r2_coluna,

                "tempo_fit":
                    np.nan,

                "tempo_transform":
                    np.nan,

                "tempo_total":
                    np.nan
            })


# ============================================================
# 10. SALVAR RESULTADOS
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


resultados_df.to_csv(

    RESULTADOS_IMPUTACAO / "resultados_knn_kfold.csv",

    index=False
)


# ============================================================
# 11. RESUMO GERAL
# ============================================================

gerais = (
    resultados_df[
        resultados_df[
            "variavel"
        ] == "GERAL"
    ]
)


resumo = (

    gerais

    .groupby(
        "percentual"
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
)


print(
    f"\n===== RESUMO KNN + {N_FOLDS}-FOLD ====="
)

print(
    resumo
)


resumo.to_csv(

    RESULTADOS_IMPUTACAO / "resumo_knn_kfold.csv"
)


# ============================================================
# 12. R² POR VARIÁVEL
# ============================================================

por_variavel = (

    resultados_df[

        resultados_df[
            "variavel"
        ] != "GERAL"

    ]
)


resumo_r2 = (

    por_variavel

    .groupby(
        [
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
)


print(
    "\n===== R² POR VARIÁVEL ====="
)

print(
    resumo_r2
)


resumo_r2.to_csv(

    RESULTADOS_IMPUTACAO / "resumo_r2_knn_kfold.csv"
)


print(
    f"\nExperimento KNN + {N_FOLDS}-Fold finalizado."
)