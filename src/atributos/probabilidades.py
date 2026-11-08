"""Converte as odds em probabilidades implícitas, sem a margem da casa.

Para cada mercado:
    1. p_bruta = 1 / odd
    2. margem  = soma(p_bruta) - 1        (lucro embutido pela casa)
    3. p       = p_bruta / soma(p_bruta)  (normalização: as probabilidades somam 1)

Exemplo (Dundee United x Aberdeen, 2005/06): odds 2,52 / 3,16 / 2,49
    -> p_bruta 0,397 / 0,316 / 0,402 (soma 1,115, margem 11,5%)
    -> p       0,3559 / 0,2838 / 0,3602 (salvas com 4 casas decimais)

Mercados com margem fora de [0%, 25%] são erros da fonte (margem negativa
significaria lucro garantido para o apostador): as probabilidades desse mercado
ficam NaN na partida, para o detector não confundir erro de dado com anomalia.

A normalização proporcional é o método mais usado; o método de Shin, que
corrige a margem maior cobrada nos azarões, fica como melhoria futura.

Entrada: data/processed/futebol_odds.csv
Saída:   data/processed/futebol_probabilidades.csv
"""
import numpy as np
import pandas as pd

from src.config import DADOS_PROCESSADOS

MARGEM_MIN, MARGEM_MAX = 0.0, 0.25

# as odds têm 2 casas decimais; mais de 4 casas na probabilidade é precisão falsa
CASAS_DECIMAIS = 4

# mercado: (colunas de odds, nomes das probabilidades, nome da margem)
MERCADOS = [
    (["odd_casa", "odd_empate", "odd_fora"],
     ["prob_casa", "prob_empate", "prob_fora"], "margem_1x2"),
    (["odd_mais25", "odd_menos25"],
     ["prob_mais25", "prob_menos25"], "margem_gols"),
    (["odd_fech_casa", "odd_fech_empate", "odd_fech_fora"],
     ["prob_fech_casa", "prob_fech_empate", "prob_fech_fora"], "margem_fech_1x2"),
    (["odd_fech_mais25", "odd_fech_menos25"],
     ["prob_fech_mais25", "prob_fech_menos25"], "margem_fech_gols"),
]


def converter(odds):
    """Probabilidades normalizadas e margem de um mercado.

    Odds ausentes ou inválidas (<= 1) deixam a linha inteira do mercado como NaN.
    """
    odds = odds.where(odds > 1)
    brutas = 1 / odds
    soma = brutas.sum(axis=1, min_count=odds.shape[1])
    return brutas.div(soma, axis=0), soma - 1


df = pd.read_csv(DADOS_PROCESSADOS / "futebol_odds.csv", low_memory=False)
print("Base com odds:", df.shape)

for colunas_odds, colunas_prob, coluna_margem in MERCADOS:
    probs, margem = converter(df[colunas_odds])
    probs.columns = colunas_prob

    invalida = (margem < MARGEM_MIN) | (margem > MARGEM_MAX)
    probs[invalida] = np.nan
    margem[invalida] = np.nan
    if invalida.any():
        print(f"\n{coluna_margem}: {invalida.sum()} partidas com margem fora de "
              f"[{MARGEM_MIN:.0%}, {MARGEM_MAX:.0%}] descartadas (erro na fonte)")

    df[colunas_prob] = probs
    df[coluna_margem] = margem

    validas = probs.notna().all(axis=1)
    soma = probs[validas].sum(axis=1)
    assert np.allclose(soma, 1), f"probabilidades de {coluna_margem} não somam 1"
    print(f"\n{coluna_margem}: {validas.sum()} partidas "
          f"({validas.mean():.1%}) | margem média {margem[validas].mean():.2%} "
          f"(mín {margem[validas].min():.2%}, máx {margem[validas].max():.2%})")
    print("  probabilidade média: " + ", ".join(
        f"{c} {probs.loc[validas, c].mean():.3f}" for c in colunas_prob))

# conferência: o favorito pelas odds vence com que frequência?
favorito = df[["prob_casa", "prob_empate", "prob_fora"]].to_numpy().argmax(axis=1)
resultado = df["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
print(f"\nFavorito do mercado acertou o resultado em {(favorito == resultado).mean():.1%} das partidas")

colunas_novas = [c for _, probs, margem in MERCADOS for c in probs + [margem]]
df[colunas_novas] = df[colunas_novas].round(CASAS_DECIMAIS)

df.to_csv(DADOS_PROCESSADOS / "futebol_probabilidades.csv", index=False)
print(f"\nBase salva em data/processed/futebol_probabilidades.csv: {df.shape}")
