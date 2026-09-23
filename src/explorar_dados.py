import pandas as pd
from pathlib import Path

#print(df_1997.shape)
#print(df_1997.columns.tolist())

#print(df_1997[['Unnamed: 10', 'Unnamed: 11']].head(10))
#print(df_1997.isnull().sum())

#for coluna in df_1997.columns:
#    if coluna.startswith("Unnamed"):
#        print(coluna)

#print(df_1997.isnull().all())

#print(df_2024.shape)
#print(df_2024.columns.tolist())

#colunas_1997 = set(df_1997.columns)
#colunas_2024 = set(df_2024.columns)

#comuns = colunas_1997 & colunas_2024

#print(comuns)

pasta_dados = Path("data/raw")

#for temporada in sorted(pasta_dados.iterdir()):
    #if temporada.is_dir():

 #       arquivo = temporada / "E0.csv"

  #      if arquivo.exists():
   #         print("Lendo:", arquivo)
    #        try:
    #            df = pd.read_csv(arquivo)
     #           print(temporada.name, "HS" in df.columns)

 #           except pd.errors.ParserError:
#                print(temporada.name, "ERRO DE ESTRUTURA")

  #          except UnicodeDecodeError:
   #             print(temporada.name, "ERRO DE CODIFICACAO")

pasta_dados = Path("data/raw")

estatisticas = [
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR"
]

for temporada in sorted(pasta_dados.iterdir()):

    if temporada.is_dir():

        arquivo = temporada / "E0.csv"

        if arquivo.exists():

            try:
                df = pd.read_csv(arquivo)

                resultados = []

                for coluna in estatisticas:
                    existe = coluna in df.columns
                    resultados.append(existe)

                print(temporada.name, all(resultados))

            except pd.errors.ParserError:
                print(temporada.name, "ERRO DE ESTRUTURA")

            except UnicodeDecodeError:
                print(temporada.name, "ERRO DE CODIFICACAO")