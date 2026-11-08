import pandas as pd
import numpy as np
import time

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.imputacao.gain_imputer import GainImputer
from src.config import DADOS_PROCESSADOS, RESULTADOS_IMPUTACAO


# ============================================================
# BASE
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


percentuais = [
    0.10,
    0.20,
    0.30,
    0.40
]


kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


resultados = []


# ============================================================
# FOLDS
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


    X_treino_original = (
        X.iloc[
            indices_treino
        ]
        .copy()
    )


    X_teste_original = (
        X.iloc[
            indices_teste
        ]
        .copy()
    )


    print(
        "Treino:",
        X_treino_original.shape
    )

    print(
        "Teste:",
        X_teste_original.shape
    )


    # ========================================================
    # PERCENTUAIS
    # ========================================================

    for percentual in percentuais:


        print(
            f"\nGAIN | "
            f"Fold {fold} | "
            f"{percentual * 100:.0f}%"
        )


        # ====================================================
        # MÁSCARA DO TREINO
        # ====================================================

        seed_treino = (
            1000
            + fold * 100
            + int(
                percentual * 100
            )
        )


        rng_treino = (
            np.random.default_rng(
                seed_treino
            )
        )


        mascara_treino = (

            rng_treino.random(
                X_treino_original.shape
            )

            < percentual
        )


        X_treino_missing = (
            X_treino_original.mask(
                mascara_treino
            )
        )


        # ====================================================
        # MÁSCARA DO TESTE
        #
        # EXATAMENTE A MESMA REGRA DO MICE E MISSFOREST.
        # ====================================================

        seed_teste = (
            42
            + fold * 100
            + int(
                percentual * 100
            )
        )


        rng_teste = (
            np.random.default_rng(
                seed_teste
            )
        )


        mascara_teste = (

            rng_teste.random(
                X_teste_original.shape
            )

            < percentual
        )


        X_teste_missing = (
            X_teste_original.mask(
                mascara_teste
            )
        )


        # ====================================================
        # CRIAR GAIN
        # ====================================================

        modelo = GainImputer(

            batch_size=128,

            hint_rate=0.7,

            alpha=50,

            iterations=2000,

            random_state=17
        )


        # ====================================================
        # TREINAR
        # ====================================================

        inicio_fit = (
            time.perf_counter()
        )


        modelo.fit(
            X_treino_missing.to_numpy()
        )


        fim_fit = (
            time.perf_counter()
        )


        tempo_fit = (
            fim_fit
            - inicio_fit
        )


        print(
            f"Tempo fit: "
            f"{tempo_fit:.4f}s"
        )


        # ====================================================
        # IMPUTAR TESTE
        # ====================================================

        inicio_transform = (
            time.perf_counter()
        )


        X_imputado = (
            modelo.transform(
                X_teste_missing.to_numpy()
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
        # MÉTRICAS GERAIS
        # ====================================================

        valores_reais = (
            X_teste_original
            .to_numpy()[
                mascara_teste
            ]
        )


        valores_imputados = (
            X_imputado[
                mascara_teste
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
            tempo_transform
        )

        print(
            "Tempo total:",
            tempo_total
        )


        resultados.append({

            "metodo":
                "GAIN",

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
        # MÉTRICAS POR VARIÁVEL
        # ====================================================

        for i, coluna in enumerate(
            colunas_numericas
        ):


            mascara_coluna = (
                mascara_teste[:, i]
            )


            real = (
                X_teste_original
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
                    "GAIN",

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


        # Liberar sessão TensorFlow
        modelo.close()


# ============================================================
# DATAFRAME
# ============================================================

resultados_df = (
    pd.DataFrame(
        resultados
    )
)


resultados_df.to_csv(

    RESULTADOS_IMPUTACAO / "resultados_gain_kfold.csv",

    index=False
)


# ============================================================
# RESUMO GERAL
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
    "\n===== RESUMO GAIN + 5-FOLD ====="
)

print(
    resumo
)


resumo.to_csv(

    RESULTADOS_IMPUTACAO / "resumo_gain_kfold.csv"
)


# ============================================================
# R²
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

    RESULTADOS_IMPUTACAO / "resumo_r2_gain_kfold.csv"
)


print(
    "\nGAIN + 5-Fold finalizado."
)