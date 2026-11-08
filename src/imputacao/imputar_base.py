"""Imputação final da base com o MICE (método vencedor da comparação).

Regras:
- Usa só as temporadas a partir de TEMPORADA_INICIAL.
- Imputa apenas partidas com até MAX_AUSENTES das 14 estatísticas faltando.
  Partidas com mais ausências (ligas que só registram o placar) ficam fora:
  preencher quase todas as estatísticas de um jogo seria inventá-lo.
- Os valores imputados são arredondados (são contagens), não negativos e
  coerentes: finalizações no alvo <= finalizações, gols do 1º tempo <= gols
  do jogo.
- As colunas `n_imputados` e `imputado` registram o que foi preenchido.

Teste com uma amostra:
    IMPUTACAO_AMOSTRA=10000 python -m src.imputacao.imputar_base
"""
import os

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.config import DADOS_PROCESSADOS
from src.imputacao.protocolo import COLUNAS

TEMPORADA_INICIAL = "2003_04"
MAX_AUSENTES = 4
AMOSTRA = os.environ.get("IMPUTACAO_AMOSTRA")
AMOSTRA = int(AMOSTRA) if AMOSTRA else None

# limites que a imputação não pode violar: (coluna, não pode passar de)
LIMITES = [("HST", "HS"), ("AST", "AS"), ("HTHG", "FTHG"), ("HTAG", "FTAG")]


df = pd.read_csv(DADOS_PROCESSADOS / "futebol_limpo.csv", low_memory=False)
print("Base limpa:", df.shape)

df = df[df["Season"] >= TEMPORADA_INICIAL].copy()
print(f"A partir de {TEMPORADA_INICIAL}:", df.shape)

ausentes = df[COLUNAS].isna().sum(axis=1)
fora = (ausentes > MAX_AUSENTES).sum()
df = df[ausentes <= MAX_AUSENTES].copy()
print(f"Fora da imputação (mais de {MAX_AUSENTES} estatísticas ausentes): {fora}")

if AMOSTRA:
    df = df.sample(n=min(AMOSTRA, len(df)), random_state=42)
    print(f"Amostra de teste: {len(df)} partidas")

df = df.reset_index(drop=True)
mascara = df[COLUNAS].isna().to_numpy()
print(f"Partidas a imputar: {mascara.any(axis=1).sum()} | células: {mascara.sum()}")

# mesmos parâmetros do MICE avaliado no k-fold
imputador = IterativeImputer(
    random_state=17,
    max_iter=5,
    tol=1e-2,
    initial_strategy="mean",
    imputation_order="random",
)
valores = imputador.fit_transform(df[COLUNAS])

# contagens: inteiras e não negativas
valores = np.clip(np.round(valores), 0, None)
imputado = pd.DataFrame(valores, columns=COLUNAS, index=df.index)

# só sobrescreve o que estava ausente
for coluna in COLUNAS:
    faltando = df[coluna].isna()
    df.loc[faltando, coluna] = imputado.loc[faltando, coluna]

# coerência entre colunas (só mexe em valores imputados)
ajustes = 0
for coluna, teto in LIMITES:
    i = COLUNAS.index(coluna)
    excede = mascara[:, i] & (df[coluna] > df[teto]).to_numpy()
    df.loc[excede, coluna] = df.loc[excede, teto]
    ajustes += excede.sum()
print("Valores ajustados por coerência:", ajustes)

df["n_imputados"] = mascara.sum(axis=1)
df["imputado"] = df["n_imputados"] > 0

assert df[COLUNAS].isna().sum().sum() == 0

# conferência: valores imputados x observados
print("\nMédia observada x imputada (colunas com imputação):")
for i, coluna in enumerate(COLUNAS):
    if mascara[:, i].any():
        obs = df.loc[~mascara[:, i], coluna].mean()
        imp = df.loc[mascara[:, i], coluna].mean()
        print(f"  {coluna:<5} observada {obs:6.2f} | imputada {imp:6.2f} | n={mascara[:, i].sum()}")

nome = "futebol_imputado_amostra.csv" if AMOSTRA else "futebol_imputado.csv"
df.to_csv(DADOS_PROCESSADOS / nome, index=False)
print(f"\nBase salva em data/processed/{nome}: {df.shape}")
