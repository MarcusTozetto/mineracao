from pathlib import Path
import pandas as pd
from src.config import DADOS_BRUTOS, DADOS_PROCESSADOS


# =========================
# CAMINHOS DO PROJETO
# =========================

pasta_dados = DADOS_BRUTOS
pasta_saida = DADOS_PROCESSADOS

# Cria data/processed caso ainda não exista
pasta_saida.mkdir(parents=True, exist_ok=True)


# =========================
# FUNÇÃO PARA LER UM CSV
# =========================

def ler_csv(arquivo):

    # Primeiro tentamos a leitura normal
    try:
        return pd.read_csv(arquivo)

    except (UnicodeDecodeError, pd.errors.ParserError):
        pass


    # Se falhar, tentamos descobrir a quantidade
    # de colunas do cabeçalho usando UTF-8
    try:
        with open(arquivo, encoding="utf-8") as f:
            cabecalho = f.readline()

        quantidade_colunas = len(cabecalho.strip().split(","))

        return pd.read_csv(
            arquivo,
            usecols=range(quantidade_colunas)
        )

    except (UnicodeDecodeError, pd.errors.ParserError):
        pass


    # Se UTF-8 continuar falhando, tentamos Latin-1
    try:
        with open(arquivo, encoding="latin1") as f:
            cabecalho = f.readline()

        quantidade_colunas = len(cabecalho.strip().split(","))

        return pd.read_csv(
            arquivo,
            encoding="latin1",
            usecols=range(quantidade_colunas)
        )

    except Exception as erro:
        print(f"ERRO: {arquivo}")
        print(erro)

        return None


# =========================
# LEITURA DOS ARQUIVOS
# =========================

dataframes = []

arquivos_lidos = 0
arquivos_com_erro = 0


for temporada in sorted(pasta_dados.iterdir()):

    # Ignora qualquer coisa que não seja uma pasta
    if not temporada.is_dir():
        continue

    # Não vamos utilizar a temporada atual
    if temporada.name == "2025_26":
        continue


    for arquivo in temporada.glob("*.csv"):

        df = ler_csv(arquivo)

        # Se não conseguimos ler o arquivo
        if df is None:
            arquivos_com_erro += 1
            continue


        # Guarda a origem de cada partida
        df["Season"] = temporada.name
        df["SourceFile"] = arquivo.name

        dataframes.append(df)

        arquivos_lidos += 1

        print(
            temporada.name,
            arquivo.name,
            df.shape
        )


# =========================
# CONCATENAÇÃO
# =========================

print("\nConcatenando os arquivos...")

df_final = pd.concat(
    dataframes,
    ignore_index=True,
    sort=False
)


# =========================
# RESULTADO
# =========================

print("\n===== RESULTADO =====")

print("Arquivos lidos:", arquivos_lidos)
print("Arquivos com erro:", arquivos_com_erro)

print("Quantidade de DataFrames:", len(dataframes))

print("Shape final:")
print(df_final.shape)

print("\nQuantidade de colunas:")
print(len(df_final.columns))


# =========================
# SALVAR
# =========================

arquivo_saida = pasta_saida / "futebol_consolidado.csv"

df_final.to_csv(
    arquivo_saida,
    index=False
)

print("\nBase salva em:")
print(arquivo_saida)