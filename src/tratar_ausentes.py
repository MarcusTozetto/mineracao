import pandas as pd

arquivo = "data/processed/futebol_limpo.csv"

df = pd.read_csv(arquivo)

print("Shape da base:", df.shape)

# 1. QUANTIDADE DE VALORES AUSENTES POR COLUNA

ausentes = df.isna().sum()

print("\nValores ausentes por coluna:")
print(
    ausentes
    .sort_values(ascending=False)
    .to_string()
)

# 2. COBERTURA DAS COLUNAS POR TEMPORADA

# cobertura_temporada = (
#     df
#     .groupby("Season")
#     .apply(lambda grupo: grupo.notna().mean() * 100)
# )

# print("\nCobertura das colunas por temporada:")
# print(cobertura_temporada.to_string())

# 3. COBERTURA DAS ESTATISTICAS POR TENPORADA

colunas_estatisticas = [
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR"
]

cobertura_estatisticas = (
    df
    .groupby("Season")[colunas_estatisticas]
    .apply(lambda grupo: grupo.notna().mean() * 100)
)

print(
    cobertura_estatisticas
    .round(2)
    .to_string()
)

# 4. COBERTURA DAS ESTATÍSTICAS POR TEMPORADA E DIVISÃO

cobertura_liga = (
    df
    .groupby(["Season", "Div"])[colunas_estatisticas]
    .apply(lambda grupo: grupo.notna().mean() * 100)
)

print(
    cobertura_liga
    .round(2)
    .to_string()
)

cobertura_parcial = (
    (cobertura_liga > 0)
    & (cobertura_liga < 100)
)

quantidade_parcial = cobertura_parcial.sum()

print("\nQuantidade de Season + Div com cobertura parcial:")
print(
    quantidade_parcial
    .sort_values(ascending=False)
    .to_string()
)

# 5. ANÁLISE DOS NaN DE HS

grupos_hs = (
    df
    .groupby(["Season", "Div"])["HS"]
    .agg(
        total="size",
        preenchidos="count"
    )
)

grupos_hs["ausentes"] = (
    grupos_hs["total"]
    - grupos_hs["preenchidos"]
)

grupos_hs["cobertura"] = (
    grupos_hs["preenchidos"]
    / grupos_hs["total"]
    * 100
)

print(grupos_hs.to_string())

# Separar grupos com ausência estrutural
grupos_estruturais_hs = grupos_hs[
    grupos_hs["cobertura"] == 0
]

# Separar grupos com ausência parcial
grupos_parciais_hs = grupos_hs[
    (grupos_hs["cobertura"] > 0)
    & (grupos_hs["cobertura"] < 100)
]

# Contar os NaN de cada situação
nan_estruturais_hs = grupos_estruturais_hs["ausentes"].sum()
nan_candidatos_hs = grupos_parciais_hs["ausentes"].sum()

print("\nHS")
print("NaN estruturais:", nan_estruturais_hs)
print("NaN candidatos à imputação:", nan_candidatos_hs)
print("NaN total:", df["HS"].isna().sum())

colunas_estatisticas = [
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY", "HR", "AR"
]

resultados = []

for coluna in colunas_estatisticas:

    grupos = (
        df
        .groupby(["Season", "Div"])[coluna]
        .agg(
            total="size",
            preenchidos="count"
        )
    )

    grupos["ausentes"] = (
        grupos["total"] - grupos["preenchidos"]
    )

    grupos["cobertura"] = (
        grupos["preenchidos"]
        / grupos["total"]
        * 100
    )

    # Grupos onde a variável nunca foi coletada
    grupos_estruturais = grupos[
        grupos["cobertura"] == 0
    ]

    # Grupos onde a variável existe, mas há alguns NaN
    grupos_parciais = grupos[
        (grupos["cobertura"] > 0)
        & (grupos["cobertura"] < 100)
    ]

    nan_estruturais = grupos_estruturais["ausentes"].sum()

    nan_candidatos = grupos_parciais["ausentes"].sum()

    nan_total = df[coluna].isna().sum()

    percentual_candidatos = (
        nan_candidatos / nan_total * 100
        if nan_total > 0
        else 0
    )

    resultados.append({
        "Variavel": coluna,
        "NaN_total": nan_total,
        "NaN_estruturais": nan_estruturais,
        "NaN_candidatos": nan_candidatos,
        "Percentual_candidatos": percentual_candidatos
    })


resumo_imputacao = pd.DataFrame(resultados)

print(
    resumo_imputacao
    .round(2)
    .to_string(index=False)
)

coberturas_parciais_hs = grupos_hs[
    (grupos_hs["cobertura"] > 0)
    & (grupos_hs["cobertura"] < 100)
]

print(
    coberturas_parciais_hs
    .sort_values("cobertura")
    .to_string()
)

caso = df[
    (df["Season"] == "2000_01")
    & (df["Div"] == "D2")
].copy()

print(
    caso[
        ["Date", "HomeTeam", "AwayTeam", "HS"]
    ].to_string(index=False)
)