"""Recorta a base ao período com odds e unifica as colunas de odds.

O Football-Data mudou o nome das colunas de média do mercado ao longo do
tempo: `BbAv*`/`BbMx*` até 2018/19 e `Avg*`/`Max*` a partir de 2019/20. Aqui
as duas versões viram uma coluna só, e as ~180 colunas de casas de apostas
individuais são descartadas.

- Período: a partir de 2005/06, primeira temporada com média de mercado e
  odds de mais/menos de 2,5 gols (antes disso só há Bet365).
- Mantém só partidas com as três odds 1X2 válidas (> 1).
- Odds de fechamento existem só a partir de 2019/20; antes ficam NaN.

Entrada: data/processed/futebol_imputado.csv
Saída:   data/processed/futebol_odds.csv
"""
import pandas as pd

from src.config import DADOS_PROCESSADOS
from src.imputacao.protocolo import COLUNAS

TEMPORADA_INICIAL = "2005_06"

IDENTIFICACAO = ["Div", "Season", "Date", "HomeTeam", "AwayTeam"]
RESULTADO = ["FTHG", "FTAG", "FTR", "HTR"]
CONTROLE = ["n_imputados", "imputado"]

# nome novo: (coluna até 2018/19, coluna a partir de 2019/20)
ODDS = {
    # média do mercado, resultado final (1X2)
    "odd_casa": ("BbAvH", "AvgH"),
    "odd_empate": ("BbAvD", "AvgD"),
    "odd_fora": ("BbAvA", "AvgA"),
    # maior odd do mercado (mede a dispersão entre casas)
    "odd_max_casa": ("BbMxH", "MaxH"),
    "odd_max_empate": ("BbMxD", "MaxD"),
    "odd_max_fora": ("BbMxA", "MaxA"),
    # média do mercado, mais/menos de 2,5 gols
    "odd_mais25": ("BbAv>2.5", "Avg>2.5"),
    "odd_menos25": ("BbAv<2.5", "Avg<2.5"),
    # fechamento (só a partir de 2019/20)
    "odd_fech_casa": (None, "AvgCH"),
    "odd_fech_empate": (None, "AvgCD"),
    "odd_fech_fora": (None, "AvgCA"),
    "odd_fech_mais25": (None, "AvgC>2.5"),
    "odd_fech_menos25": (None, "AvgC<2.5"),
}


df = pd.read_csv(DADOS_PROCESSADOS / "futebol_imputado.csv", low_memory=False)
print("Base imputada:", df.shape)

df = df[df["Season"] >= TEMPORADA_INICIAL]
print(f"A partir de {TEMPORADA_INICIAL}:", df.shape)

odds = pd.DataFrame(index=df.index)
for nome, (antiga, nova) in ODDS.items():
    serie = pd.to_numeric(df[nova], errors="coerce")
    if antiga:
        serie = pd.to_numeric(df[antiga], errors="coerce").fillna(serie)
    odds[nome] = serie

base = pd.concat([df[IDENTIFICACAO + RESULTADO + COLUNAS + CONTROLE], odds], axis=1)

validas = (base[["odd_casa", "odd_empate", "odd_fora"]] > 1).all(axis=1)
print("Partidas sem odds 1X2 válidas (removidas):", (~validas).sum())
base = base[validas].reset_index(drop=True)

print("\nCobertura das odds (%):")
print((base[list(ODDS)].notna().mean() * 100).round(1).to_string())

base.to_csv(DADOS_PROCESSADOS / "futebol_odds.csv", index=False)
print(f"\nBase salva em data/processed/futebol_odds.csv: {base.shape}")
