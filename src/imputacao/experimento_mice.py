import pandas as pd
import numpy as np
import time

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
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


# ============================================================
# 2. VARIÁVEIS NUMÉRICAS
# ============================================================

colunas_numericas = [
    "HTHG", "HTAG",
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR"
]


# ============================================================
# 3. BASE COMPLETA
# ============================================================

# Agora NÃO usamos sample(20000).
# Vamos utilizar todas as observações que possuem
# as 14 variáveis completas.

X = (
    df[colunas_numericas]
    .dropna()
    .reset_index(drop=True)
    .copy()
)

print("Base completa:", X.shape)


# ============================================================
# 4. CONFIGURAÇÕES
# ============================================================

percentuais = [
    0.10,
    0.20,
    0.30,
    0.40
]


# ============================================================
# 5. CONFIGURAR 5-FOLD
# ============================================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Guardaremos todos os resultados aqui
resultados = []


# ============================================================
# 6. LOOP DOS FOLDS
# ============================================================

for fold, (indices_treino, indices_teste) in enumerate(
    kfold.split(X),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)


    # --------------------------------------------------------
    # Separar treino e teste
    # --------------------------------------------------------

    X_treino = (
        X.iloc[indices_treino]
        .copy()
    )

    X_teste = (
        X.iloc[indices_teste]
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
    # 7. TREINAR O MICE
    # ========================================================

    # Mesmos parâmetros utilizados anteriormente.
    imputador = IterativeImputer(
        random_state=17,
        max_iter=5,
        tol=1e-2,
        initial_strategy="mean",
        imputation_order="random"
    )


    inicio_fit = time.perf_counter()

    imputador.fit(
        X_treino
    )

    fim_fit = time.perf_counter()


    tempo_fit = (
        fim_fit - inicio_fit
    )


    print(
        f"Tempo de treinamento: "
        f"{tempo_fit:.4f} segundos"
    )


    # ========================================================
    # 8. TESTAR DIFERENTES PERCENTUAIS DE AUSÊNCIA
    # ========================================================

    for percentual in percentuais:

        print(
            f"\nMICE | "
            f"Fold {fold} | "
            f"{percentual * 100:.0f}%"
        )


        # ----------------------------------------------------
        # Seed da máscara
        # ----------------------------------------------------
        #
        # IMPORTANTE:
        #
        # Essa fórmula será exatamente a mesma nos outros
        # algoritmos.
        #
        # Assim:
        #
        # Fold 1 + 10% do MICE
        #
        # terá a mesma máscara que:
        #
        # Fold 1 + 10% do MissForest/KNN/GAIN.
        # ----------------------------------------------------

        seed_mascara = (
            42
            + fold * 100
            + int(percentual * 100)
        )


        rng = np.random.default_rng(
            seed_mascara
        )


        # ----------------------------------------------------
        # Criar máscara MCAR SOMENTE NO TESTE
        # ----------------------------------------------------

        mascara = (
            rng.random(
                X_teste.shape
            )
            < percentual
        )


        # ----------------------------------------------------
        # Esconder os valores
        # ----------------------------------------------------

        X_teste_missing = (
            X_teste.mask(
                mascara
            )
        )


        # ====================================================
        # 9. IMPUTAR O TESTE
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


        # ----------------------------------------------------
        # Tempo total
        # ----------------------------------------------------

        tempo_total = (
            tempo_fit
            + tempo_transform
        )


        # ====================================================
        # 10. VALORES QUE FORAM ESCONDIDOS
        # ====================================================

        valores_reais = (
            X_teste
            .to_numpy()[
                mascara
            ]
        )


        valores_imputados = (
            X_imputado[
                mascara
            ]
        )


        # ====================================================
        # 11. MÉTRICAS GERAIS
        # ====================================================

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


        # ====================================================
        # 12. SALVAR RESULTADO GERAL
        # ====================================================

        resultados.append({

            "metodo":
                "MICE",

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
        # 13. MÉTRICAS POR VARIÁVEL
        # ====================================================

        for i, coluna in enumerate(
            colunas_numericas
        ):

            # Quais células dessa coluna
            # foram artificialmente escondidas?
            mascara_coluna = (
                mascara[:, i]
            )


            # Valores verdadeiros
            real = (
                X_teste.iloc[:, i]
                .to_numpy()[
                    mascara_coluna
                ]
            )


            # Valores produzidos pelo MICE
            imputado = (
                X_imputado[:, i][
                    mascara_coluna
                ]
            )


            # -----------------------------------------------
            # MAE
            # -----------------------------------------------

            mae_coluna = (
                mean_absolute_error(
                    real,
                    imputado
                )
            )


            # -----------------------------------------------
            # MSE
            # -----------------------------------------------

            mse_coluna = (
                mean_squared_error(
                    real,
                    imputado
                )
            )


            # -----------------------------------------------
            # RMSE
            # -----------------------------------------------

            rmse_coluna = (
                np.sqrt(
                    mse_coluna
                )
            )


            # -----------------------------------------------
            # R²
            # -----------------------------------------------

            r2_coluna = (
                r2_score(
                    real,
                    imputado
                )
            )


            resultados.append({

                "metodo":
                    "MICE",

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
# 14. CRIAR DATAFRAME
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


# ============================================================
# 15. SALVAR RESULTADOS COMPLETOS
# ============================================================

resultados_df.to_csv(
    RESULTADOS_IMPUTACAO / "resultados_mice_kfold.csv",
    index=False
)


# ============================================================
# 16. RESULTADOS GERAIS
# ============================================================

gerais = (
    resultados_df[
        resultados_df[
            "variavel"
        ] == "GERAL"
    ]
)


# ============================================================
# 17. RESUMO DAS MÉTRICAS
# ============================================================

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
    "\n===== RESUMO MICE + 5-FOLD ====="
)

print(
    resumo
)


resumo.to_csv(
    RESULTADOS_IMPUTACAO / "resumo_mice_kfold.csv"
)


# ============================================================
# 18. R² POR VARIÁVEL
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
    RESULTADOS_IMPUTACAO / "resumo_r2_mice_kfold.csv"
)


print(
    "\nExperimento MICE + 5-Fold finalizado."
)