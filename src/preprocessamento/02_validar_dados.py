import pandas as pd
from src.config import DADOS_PROCESSADOS


# ============================================================
# 1. CARREGAMENTO DA BASE CONSOLIDADA
# ============================================================

arquivo = DADOS_PROCESSADOS / "futebol_consolidado.csv"

df = pd.read_csv(arquivo)

print("Shape original:", df.shape)


# ============================================================
# 2. CRIAÇÃO DA BASE DE TRABALHO
# ============================================================

# Criamos uma cópia para não alterar o DataFrame original.
df_limpo = df.copy()


# ============================================================
# 3. REMOÇÃO DAS COLUNAS UNNAMED
# ============================================================

# Localiza colunas que não possuem nome no CSV original.
colunas_unnamed = [
    coluna
    for coluna in df_limpo.columns
    if coluna.startswith("Unnamed")
]

# Essas colunas já foram investigadas.
# 61 eram completamente vazias e 3 possuíam valores isolados
# sem identificação do atributo no arquivo original.
df_limpo = df_limpo.drop(columns=colunas_unnamed)

print("Colunas Unnamed removidas:", len(colunas_unnamed))
print("Shape após remover Unnamed:", df_limpo.shape)


# ============================================================
# 4. REMOÇÃO DE LINHAS QUE NÃO REPRESENTAM PARTIDAS
# ============================================================

# Uma linha é considerada sem partida quando não possui
# data, time mandante nem time visitante.
colunas_partida = ["Date", "HomeTeam", "AwayTeam"]

linhas_sem_partida = (
    df_limpo[colunas_partida]
    .isna()
    .all(axis=1)
)

print(
    "Linhas sem dados de partida:",
    linhas_sem_partida.sum()
)

# Após investigação, verificamos que essas linhas eram
# registros vazios ou inválidos.
df_limpo = df_limpo.loc[~linhas_sem_partida].copy()

print(
    "Shape após remover linhas inválidas:",
    df_limpo.shape
)


# ============================================================
# 5. PADRONIZAÇÃO DOS NOMES DOS TIMES
# ============================================================

# Alguns arquivos antigos da liga grega utilizam:
# HT = HomeTeam
# AT = AwayTeam
#
# Já verificamos que HT/HomeTeam e AT/AwayTeam
# nunca possuem valores simultaneamente.

df_limpo["HomeTeam"] = (
    df_limpo["HomeTeam"]
    .fillna(df_limpo["HT"])
)

df_limpo["AwayTeam"] = (
    df_limpo["AwayTeam"]
    .fillna(df_limpo["AT"])
)

# Depois da transferência, HT e AT se tornam redundantes.
df_limpo = df_limpo.drop(columns=["HT", "AT"])

print(
    "HomeTeam ausentes:",
    df_limpo["HomeTeam"].isna().sum()
)

print(
    "AwayTeam ausentes:",
    df_limpo["AwayTeam"].isna().sum()
)

print(
    "Shape após padronizar times:",
    df_limpo.shape
)


# ============================================================
# 6. VERIFICAÇÃO DE DUPLICATAS
# ============================================================

# Definimos uma partida pela combinação:
# temporada + divisão + data + mandante + visitante.
chave_partida = [
    "Season",
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam"
]

duplicadas = df_limpo.duplicated(
    subset=chave_partida,
    keep=False
)

print(
    "Linhas envolvidas em possíveis duplicatas:",
    duplicadas.sum()
)


# ============================================================
# 7. TIPOS DAS COLUNAS
# ============================================================

# Próxima etapa da nossa validação.
print("\nTipos das colunas:")
print(df_limpo.dtypes.to_string())

colunas_object = df_limpo.select_dtypes(
    include=["object"]
).columns

print("\nColunas do tipo object:")
print(colunas_object.tolist())

print("\nQuantidade de colunas object:")
print(len(colunas_object))

print("\nQuantidade de colunas por tipo:")
print(df_limpo.dtypes.value_counts())

for coluna in colunas_object:
    print(
        coluna,
        "->",
        df_limpo[coluna].notna().sum(),
        "valores preenchidos"
    )

# ============================================================
# 8. INVESTIGAÇÃO DAS COLUNAS OBJECT
# ============================================================

colunas_object = df_limpo.select_dtypes(
    include=["object"]
).columns

for coluna in colunas_object:

    print("\nCOLUNA:", coluna)

    tipos = (
        df_limpo[coluna]
        .dropna()
        .apply(type)
        .value_counts()
    )

    print(tipos)    


colunas_mistas = [
    "BbAH",
    "BbAHh",
    "PSCH",
    "B365CH",
    "BWCA",
    "AvgCAHH",
    "1XBH"
]

for coluna in colunas_mistas:

    # Seleciona apenas os valores que foram lidos como texto.
    valores_texto = df_limpo.loc[
        df_limpo[coluna].apply(lambda valor: isinstance(valor, str)),
        coluna
    ]

    # Tenta converter esses textos para número.
    conversao = pd.to_numeric(
        valores_texto,
        errors="coerce"
    )

    # Se virou NaN, significa que o texto não era um número válido.
    invalidos = valores_texto[conversao.isna()]

    print("\nCOLUNA:", coluna)
    print("Strings:", len(valores_texto))
    print("Strings não numéricas:", len(invalidos))

    if len(invalidos) > 0:
        print(invalidos.value_counts().head(20))

# ============================================================
# 9. LOCALIZAÇÃO DOS VALORES NÃO NUMÉRICOS
# ============================================================

colunas_mistas = [
    "BbAH",
    "BbAHh",
    "PSCH",
    "B365CH",
    "BWCA",
    "AvgCAHH",
    "1XBH"
]

for coluna in colunas_mistas:

    valores_convertidos = pd.to_numeric(
        df_limpo[coluna],
        errors="coerce"
    )

    # Valor original existe, mas não conseguiu virar número.
    mascara_invalida = (
        df_limpo[coluna].notna()
        & valores_convertidos.isna()
    )

    if mascara_invalida.sum() > 0:

        print("\nCOLUNA:", coluna)

        print(
            df_limpo.loc[
                mascara_invalida,
                [
                    "Season",
                    "SourceFile",
                    "Div",
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    coluna
                ]
            ].to_string(index=False)
        )

# ============================================================
# 10. INSPEÇÃO DAS LINHAS COM VALORES SUSPEITOS
# ============================================================

colunas_suspeitas = [
    "BbAH",
    "BbAHh",
    "B365CH",
    "BWCA",
    "AvgCAHH",
    "1XBH"
]

for coluna in colunas_suspeitas:

    valores_convertidos = pd.to_numeric(
        df_limpo[coluna],
        errors="coerce"
    )

    mascara = (
        df_limpo[coluna].notna()
        & valores_convertidos.isna()
    )

    if mascara.sum() > 0:

        indices = df_limpo.index[mascara]

        for indice in indices:

            print("\n" + "=" * 70)
            print("COLUNA:", coluna)
            print("ÍNDICE:", indice)
            print("=" * 70)

            # Posição da coluna problemática
            posicao = df_limpo.columns.get_loc(coluna)

            # Mostra algumas colunas antes e depois dela
            inicio = max(0, posicao - 4)
            fim = min(
                len(df_limpo.columns),
                posicao + 5
            )

            colunas_vizinhas = (
                df_limpo.columns[inicio:fim]
            )

            print(
                df_limpo.loc[
                    indice,
                    colunas_vizinhas
                ]
            )


def converter_handicap(valor):

    if pd.isna(valor):
        return valor

    # Caso seja um handicap dividido, como "-0.5,-1"
    if isinstance(valor, str) and "," in valor:

        partes = valor.split(",")

        numeros = [
            float(parte)
            for parte in partes
        ]

        return sum(numeros) / len(numeros)

    return valor

print("\nTipos após tratamento:")

colunas_numericas = [
    "BbAH",
    "BbAHh",
    "PSCH",
    "B365CH",
    "BWCA",
    "AvgCAHH",
    "1XBH"
]

for coluna in colunas_numericas:

    df_limpo[coluna] = pd.to_numeric(
        df_limpo[coluna],
        errors="coerce"
    )

for coluna in colunas_numericas:
    print(
        coluna,
        "->",
        df_limpo[coluna].dtype
    )

print("\nQuantidade de colunas por tipo:")
print(df_limpo.dtypes.value_counts())

# ============================================================
# 10. ANÁLISE DA COLUNA DATE
# ============================================================

print("\nExemplos de datas:")

print(
    df_limpo[
        ["Season", "SourceFile", "Date"]
    ]
    .drop_duplicates()
    .head(30)
    .to_string(index=False)
)

tamanho_datas = (
    df_limpo["Date"]
    .str.len()
    .value_counts()
    .sort_index()
)

print("\nTamanhos encontrados na coluna Date:")
print(tamanho_datas)

# ============================================================
# 11. VALIDAÇÃO DAS DATAS
# ============================================================

datas_convertidas = pd.to_datetime(
    df_limpo["Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

datas_invalidas = datas_convertidas.isna()

print(
    "Datas que não puderam ser convertidas:",
    datas_invalidas.sum()
)

print(
    df_limpo.loc[
        datas_invalidas,
        ["Season", "SourceFile", "Date"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

# ============================================================
# 12. CONVERSÃO DA COLUNA DATE
# ============================================================

df_limpo["Date"] = datas_convertidas

print("Tipo de Date:", df_limpo["Date"].dtype)

# ============================================================
# 13. QUANTIDADE DE PARTIDAS POR TEMPORADA E LIGA
# ============================================================

partidas_por_liga = (
    df_limpo
    .groupby(["Season", "Div"])
    .size()
    .reset_index(name="Quantidade")
)

print(partidas_por_liga.to_string(index=False))

# ============================================================
# 14. VALORES IMPOSSÍVEIS
# ============================================================

colunas_gols = [
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG"
]

for coluna in colunas_gols:
    negativos = (df_limpo[coluna] < 0).sum()

    print(
        coluna,
        "-> negativos:",
        negativos,
        "| mínimo:",
        df_limpo[coluna].min(),
        "| máximo:",
        df_limpo[coluna].max()
    )

# Verifica inconsistências entre intervalo e final

erro_gols_casa = df_limpo["HTHG"] > df_limpo["FTHG"]
erro_gols_fora = df_limpo["HTAG"] > df_limpo["FTAG"]

print(
    "HTHG maior que FTHG:",
    erro_gols_casa.sum()
)

print(
    "HTAG maior que FTAG:",
    erro_gols_fora.sum()
)

# Verifica se o resultado corresponde ao placar final

resultado_esperado = pd.Series(index=df_limpo.index, dtype="string")

resultado_esperado[df_limpo["FTHG"] > df_limpo["FTAG"]] = "H"
resultado_esperado[df_limpo["FTHG"] < df_limpo["FTAG"]] = "A"
resultado_esperado[df_limpo["FTHG"] == df_limpo["FTAG"]] = "D"

resultado_incorreto = (
    df_limpo["FTR"].notna()
    & (df_limpo["FTR"] != resultado_esperado)
)

print(
    "Resultados incompatíveis com o placar:",
    resultado_incorreto.sum()
)

# Verifica partidas sem resultado final

ftr_ausente = df_limpo["FTR"].isna()

print("FTR ausentes:", ftr_ausente.sum())

print(
    df_limpo.loc[
        ftr_ausente,
        [
            "Season",
            "Div",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR"
        ]
    ].to_string(index=False)
)

# Identifica partidas sem placar final

partidas_sem_resultado = (
    df_limpo["FTHG"].isna()
    & df_limpo["FTAG"].isna()
    & df_limpo["FTR"].isna()
)

print(
    "Partidas sem resultado:",
    partidas_sem_resultado.sum()
)

# ============================================================
# 15. REMOÇÃO DE PARTIDAS SEM RESULTADO
# ============================================================

# Remove partidas que não possuem placar nem resultado final.
df_limpo = df_limpo.loc[~partidas_sem_resultado].copy()

print(
    "Partidas removidas:",
    partidas_sem_resultado.sum()
)

print(
    "Shape após remover partidas sem resultado:",
    df_limpo.shape
)


# ============================================================
# 16. SALVAMENTO DA BASE LIMPA
# ============================================================

arquivo_saida = DADOS_PROCESSADOS / "futebol_limpo.csv"

df_limpo.to_csv(
    arquivo_saida,
    index=False
)

print("\nBase limpa salva em:", arquivo_saida)
print("Shape final:", df_limpo.shape)