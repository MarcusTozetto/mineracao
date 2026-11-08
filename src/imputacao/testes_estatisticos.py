"""Testes estatísticos da comparação de imputação.

Unidade de comparação (bloco): uma variável em um fold. Com 14 variáveis e
5 folds são 70 blocos pareados por método e percentual, em vez de 5. Com só
5 folds o menor p possível do Wilcoxon bicaudal é 0,0625, então nenhuma
diferença conseguiria ser significativa a 0,05 nem a 0,01.

Testes, para cada métrica (MAE, RMSE, R²) e cada percentual de ausência:

1. Friedman: existe diferença entre os métodos?
   Ranqueia os métodos dentro de cada bloco (1 = menor erro) e testa se os
   ranks médios diferem. É o teste padrão para comparar vários métodos
   (Demšar, 2006).

2. Wilcoxon bicaudal entre todos os pares, com correção de Holm.
   H0: os dois métodos erram igual. Exploratório: não presume quem é melhor.

3. Wilcoxon unicaudal do método de referência (MICE) contra cada outro, com
   correção de Holm. H1: o MICE erra MENOS. A direção é definida antes de
   olhar os dados (o MICE foi o método escolhido), por isso o teste unicaudal
   é legítimo aqui; escolher a direção depois de ver o resultado não seria.

Os resultados saem para α = 0,05 e α = 0,01, com tamanho de efeito:
    vitorias   fração dos blocos em que o método A errou menos que o B
    r_rb       correlação rank-bisserial (-1 a 1): magnitude da diferença,
               independente do tamanho da amostra (|r| >= 0,5 é grande)

Limitação: blocos do mesmo fold usam os mesmos dados, então não são
totalmente independentes, e os p-valores tendem a ser otimistas. Por isso
os tamanhos de efeito devem ser lidos junto com os p-valores. A solução
completa é repetir a validação cruzada (por exemplo, 10 x 5 folds).

Entrada: results/imputacao/resultados_<metodo>_kfold.csv
Saídas:  results/imputacao/friedman_kfold.csv
         results/imputacao/wilcoxon_kfold.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from src.config import RESULTADOS_IMPUTACAO

NOMES = ["mice", "missforest", "gain", "knn", "mediana"]
ORDEM = ["MICE", "MissForest", "GAIN", "KNN", "Mediana"]
REFERENCIA = "MICE"
METRICAS = {"MAE": "menor", "RMSE": "menor", "R2": "maior"}
ALFAS = [0.05, 0.01]


def carregar():
    dfs = []
    for nome in NOMES:
        arquivo = RESULTADOS_IMPUTACAO / f"resultados_{nome}_kfold.csv"
        if arquivo.exists():
            dfs.append(pd.read_csv(arquivo))
        else:
            print("Sem resultados (ignorado):", arquivo.name)
    df = pd.concat(dfs, ignore_index=True)
    return df[df["variavel"] != "GERAL"]


def rank_bisserial(a, b):
    """Efeito do Wilcoxon pareado: (soma dos ranks + - soma dos ranks -) / total."""
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


df = carregar()
metodos = [m for m in ORDEM if m in set(df["metodo"])]
print("Métodos:", metodos)

friedman, pares = [], []

for metrica, sentido in METRICAS.items():
    for percentual in sorted(df["percentual"].unique()):
        blocos = (df[df["percentual"] == percentual]
                  .pivot_table(index=["fold", "variavel"], columns="metodo",
                               values=metrica)[metodos]
                  .dropna())
        # "erro": quanto menor, melhor (R² é invertido)
        erro = blocos if sentido == "menor" else -blocos
        n = len(erro)

        # 1. Friedman + ranks médios
        estat, p = friedmanchisquare(*[erro[m] for m in metodos])
        ranks = erro.rank(axis=1).mean()
        linha = {"metrica": metrica, "percentual": percentual, "blocos": n,
                 "estatistica": estat, "p": p}
        linha.update({f"rank_{m}": ranks[m] for m in metodos})
        friedman.append(linha)

        # 2 e 3. Wilcoxon pareado
        for tipo, lista in [
            ("bicaudal", list(combinations(metodos, 2))),
            ("unicaudal", [(REFERENCIA, m) for m in metodos if m != REFERENCIA]),
        ]:
            grupo = []
            for a, b in lista:
                alternativa = "two-sided" if tipo == "bicaudal" else "less"
                _, p_par = wilcoxon(erro[a], erro[b], alternative=alternativa)
                grupo.append({
                    "metrica": metrica, "percentual": percentual, "teste": tipo,
                    "metodo_a": a, "metodo_b": b, "blocos": n,
                    "media_a": blocos[a].mean(), "media_b": blocos[b].mean(),
                    "vitorias_a": (erro[a] < erro[b]).mean(),
                    "r_rb": -rank_bisserial(erro[a].to_numpy(), erro[b].to_numpy()),
                    "p": p_par,
                })
            p_holm = multipletests([g["p"] for g in grupo], method="holm")[1]
            for g, ph in zip(grupo, p_holm):
                g["p_holm"] = ph
                for alfa in ALFAS:
                    g[f"signif_{alfa}"] = ph < alfa
            pares += grupo

friedman = pd.DataFrame(friedman)
pares = pd.DataFrame(pares)
friedman.to_csv(RESULTADOS_IMPUTACAO / "friedman_kfold.csv", index=False)
pares.to_csv(RESULTADOS_IMPUTACAO / "wilcoxon_kfold.csv", index=False)

# ------------------------------------------------------------------
# Resumo na tela
# ------------------------------------------------------------------
pd.set_option("display.width", 160)
print("\n===== FRIEDMAN — rank médio (1 = melhor) =====")
cols = ["metrica", "percentual", "p"] + [f"rank_{m}" for m in metodos]
print(friedman[cols].round(4).to_string(index=False))

print(f"\n===== UNICAUDAL: {REFERENCIA} erra menos que ... (Holm) =====")
uni = pares[pares["teste"] == "unicaudal"]
print(uni[["metrica", "percentual", "metodo_b", "vitorias_a", "r_rb", "p_holm",
           "signif_0.05", "signif_0.01"]].round(4).to_string(index=False))

print("\n===== BICAUDAL: pares NÃO significativos a 0,01 (Holm) =====")
bi = pares[(pares["teste"] == "bicaudal") & ~pares["signif_0.01"]]
if len(bi):
    print(bi[["metrica", "percentual", "metodo_a", "metodo_b", "vitorias_a", "r_rb",
              "p_holm"]].round(4).to_string(index=False))
else:
    print("(todos os pares diferem a 0,01)")

print("\nResultados salvos em results/imputacao/friedman_kfold.csv e wilcoxon_kfold.csv")
