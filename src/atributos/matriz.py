"""Matriz final para os algoritmos de detecção.

Princípio: o detector aponta o que é RARO na matriz. Então a matriz só leva
atributos em que "raro" significa "estranho", isto é, desvios em relação ao
que se esperava do jogo. Atributos de expectativa ou contexto (probabilidades,
favoritismo, forma, reta final, domínio bruto) ficam fora: um jogo entre um
gigante e um pequeno é raro, mas não é estranho. Eles continuam em
futebol_atributos.csv para interpretar as partidas apontadas.

Atributos da matriz
    Contínuos (z-score dentro de cada liga e temporada):
        surpresa_resultado   o resultado contrariou o mercado?
        surpresa_gols        o lado de 2,5 gols contrariou o mercado?
        desvio_gols          gols reais - gols esperados pelo mercado
        dispersao_odds       casas discordando sobre o resultado ocorrido
                             (limitada ao percentil 99,9: valores acima são
                             odds máximas digitadas erradas na fonte)
        margem_1x2           margem da casa: incerteza do mercado
        desencontro_dominio  domínio real - domínio esperado pelo mercado
        conversao            gols / finalizações no alvo
        prop_gols_2t         fração dos gols no 2º tempo
        perda_vantagem       saldo perdido no 2º tempo por quem vencia no
                             intervalo (versão contínua da virada)
        total_faltas
        indice_disciplina    amarelos + 2,5 x vermelhos (um índice só, para a
                             disciplina não pesar mais que o mercado)
        dif_cartoes          desequilíbrio de cartões entre as equipes

    Sem atributos binários (0/1): o Isolation Forest isola um evento raro de
    0/1 em poucos cortes, qualquer que seja a escala, e eles dominariam o
    detector. vitoria_dominado e virada/vantagem_perdida foram substituídos
    pelas versões contínuas desencontro_dominio e perda_vantagem.

Vazios: depois do z-score viram 0 (= média da liga). A coluna
`sem_historico` foi substituída por `sem_odds_gols` (jogos sem odds de 2,5 gols).

Entrada: data/processed/futebol_atributos.csv
Saídas:  data/processed/futebol_matriz.csv
         results/atributos/correlacao_atributos.csv
"""
import numpy as np
import pandas as pd

from src.config import DADOS_PROCESSADOS, RESULTADOS

CONTINUOS = [
    "surpresa_resultado", "surpresa_gols", "desvio_gols", "dispersao_odds",
    "margem_1x2", "desencontro_dominio", "conversao", "prop_gols_2t",
    "perda_vantagem", "total_faltas", "indice_disciplina", "dif_cartoes",
]
BINARIOS = []

# limitados ao percentil indicado antes da normalização (erros da fonte)
LIMITAR = {"dispersao_odds": 0.999}

IDENTIFICACAO = ["Div", "Season", "Date", "HomeTeam", "AwayTeam",
                 "FTHG", "FTAG", "FTR", "imputado"]
LIMITE_CORRELACAO = 0.8
CASAS_DECIMAIS = 4

df = pd.read_csv(DADOS_PROCESSADOS / "futebol_atributos.csv", low_memory=False)
print("Base de atributos:", df.shape)

colunas = CONTINUOS + BINARIOS
X = df[colunas].astype(float)

# --- 1. redundância entre os atributos escolhidos ---
correlacao = X.corr(method="spearman")
pasta = RESULTADOS / "atributos"
pasta.mkdir(parents=True, exist_ok=True)
correlacao.round(3).to_csv(pasta / "correlacao_atributos.csv")

altos = [(a, b, round(correlacao.loc[a, b], 2))
         for i, a in enumerate(colunas) for b in colunas[i + 1:]
         if abs(correlacao.loc[a, b]) >= LIMITE_CORRELACAO]
assert not altos, f"atributos redundantes na matriz: {altos}"
maior = correlacao.where(~np.eye(len(colunas), dtype=bool)).abs().stack().idxmax()
print(f"Maior correlação entre atributos: {maior} = {correlacao.loc[maior]:.2f} "
      f"(limite {LIMITE_CORRELACAO})")

# --- 2. limita valores extremos que são erro da fonte ---
for coluna, quantil in LIMITAR.items():
    teto = X[coluna].quantile(quantil)
    acima = (X[coluna] > teto).sum()
    X[coluna] = X[coluna].clip(upper=teto)
    print(f"{coluna}: {acima} valores acima do percentil {quantil:.1%} "
          f"limitados a {teto:.3f}")

# --- 3. z-score dos contínuos dentro de cada liga e temporada ---
grupos = [df["Div"], df["Season"]]
C = X[CONTINUOS]
media = C.groupby(grupos).transform("mean")
desvio = C.groupby(grupos).transform("std")
Z = (C - media) / desvio.replace(0, np.nan)
Z = Z.where(desvio > 0, 0.0)  # atributo constante na liga-temporada

vazios = Z.isna().sum()
print("\nValores vazios preenchidos com 0 (média da liga):")
print(vazios[vazios > 0].to_string())
Z = Z.fillna(0.0)

matriz = pd.concat([
    df[IDENTIFICACAO],
    df["prob_mais25"].isna().astype(int).rename("sem_odds_gols"),
    Z.round(CASAS_DECIMAIS).add_prefix("z_"),
    X[BINARIOS].astype(int).add_prefix("b_"),
], axis=1)

atributos = [c for c in matriz.columns if c.startswith(("z_", "b_"))]
assert matriz[atributos].isna().sum().sum() == 0

print("\nConferência (média ~0, desvio ~1):")
print(matriz[atributos].agg(["mean", "std"]).T.round(3).to_string())

matriz.to_csv(DADOS_PROCESSADOS / "futebol_matriz.csv", index=False)
print(f"\nMatriz salva em data/processed/futebol_matriz.csv: {matriz.shape} "
      f"({len(atributos)} atributos)")
