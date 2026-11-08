import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import FIGURAS, RESULTADOS_IMPUTACAO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = RESULTADOS_IMPUTACAO
PASTA_GRAFICOS = FIGURAS

PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120
})


# ============================================================
# CARREGAR RESULTADOS
# ============================================================

resumo = pd.read_csv(
    PASTA_DADOS / "comparacao_metodos_kfold.csv"
)

r2 = pd.read_csv(
    PASTA_DADOS / "comparacao_r2_kfold.csv"
)


# Ordem que queremos manter nos gráficos
ordem = ["MICE", "MissForest", "KNN", "GAIN", "TabPFN", "Mediana"]

metodos = [
    m for m in ordem
    if m in set(resumo["metodo"])
]


# ============================================================
# FUNÇÃO AUXILIAR
# ============================================================

def dados_metodo(df, metodo):
    return (
        df[df["metodo"] == metodo]
        .sort_values("percentual")
    )


# ============================================================
# GRÁFICO 1
# MAE × PERCENTUAL DE AUSÊNCIA
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))


for metodo in metodos:

    dados = dados_metodo(
        resumo,
        metodo
    )

    ax.plot(
        dados["percentual"],
        dados["MAE_medio"],
        marker="o",
        linewidth=2,
        label=metodo
    )


ax.set_title(
    "MAE por percentual de dados ausentes"
)

ax.set_xlabel(
    "Dados ausentes (%)"
)

ax.set_ylabel(
    "MAE"
)

ax.set_xticks(
    [10, 20, 30, 40]
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PASTA_GRAFICOS / "mae_por_percentual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# GRÁFICO 2
# RMSE × PERCENTUAL DE AUSÊNCIA
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))


for metodo in metodos:

    dados = dados_metodo(
        resumo,
        metodo
    )

    ax.plot(
        dados["percentual"],
        dados["RMSE_medio"],
        marker="o",
        linewidth=2,
        label=metodo
    )


ax.set_title(
    "RMSE por percentual de dados ausentes"
)

ax.set_xlabel(
    "Dados ausentes (%)"
)

ax.set_ylabel(
    "RMSE"
)

ax.set_xticks(
    [10, 20, 30, 40]
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PASTA_GRAFICOS / "rmse_por_percentual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# GRÁFICO 3
# TEMPO × PERCENTUAL DE AUSÊNCIA
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))


for metodo in metodos:

    dados = dados_metodo(
        resumo,
        metodo
    )

    ax.plot(
        dados["percentual"],
        dados["tempo_total_medio"],
        marker="o",
        linewidth=2,
        label=metodo
    )


ax.set_title(
    "Tempo médio de execução"
)

ax.set_xlabel(
    "Dados ausentes (%)"
)

ax.set_ylabel(
    "Tempo médio (s)"
)

ax.set_xticks(
    [10, 20, 30, 40]
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PASTA_GRAFICOS / "tempo_por_percentual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# GRÁFICO 4
# MAE COM DESVIO PADRÃO
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))


for metodo in metodos:

    dados = dados_metodo(
        resumo,
        metodo
    )

    ax.errorbar(
        dados["percentual"],
        dados["MAE_medio"],
        yerr=dados["MAE_desvio"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=metodo
    )


ax.set_title(
    "MAE médio e desvio padrão entre os folds"
)

ax.set_xlabel(
    "Dados ausentes (%)"
)

ax.set_ylabel(
    "MAE"
)

ax.set_xticks(
    [10, 20, 30, 40]
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PASTA_GRAFICOS / "mae_desvio_padrao.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# GRÁFICO 5
# R² POR VARIÁVEL COM 10% DE AUSÊNCIA
# ============================================================

r2_10 = r2[
    r2["percentual"] == 10
].copy()


tabela_r2 = r2_10.pivot(
    index="variavel",
    columns="metodo",
    values="R2_medio"
)


# Garantir ordem dos métodos
tabela_r2 = tabela_r2[
    metodos
]


fig, ax = plt.subplots(
    figsize=(12, 6)
)


tabela_r2.plot(
    kind="bar",
    ax=ax
)


ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "R² por variável com 10% de dados ausentes"
)

ax.set_xlabel(
    "Variável"
)

ax.set_ylabel(
    "R² médio"
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.legend(
    title="Método"
)

ax.grid(
    axis="y",
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    PASTA_GRAFICOS / "r2_por_variavel_10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print(
    "Gráficos gerados com sucesso!"
)

print(
    f"Pasta: {PASTA_GRAFICOS.resolve()}"
)