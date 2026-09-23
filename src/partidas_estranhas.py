from pathlib import Path


def procurar_partida(temporada, arquivo, time_casa, time_fora):

    caminho = Path("data/raw") / temporada / arquivo

    with open(
        caminho,
        encoding="latin1",
        errors="replace"
    ) as f:

        linhas = f.readlines()

    print("\nArquivo:", caminho)

    # Cabeçalho
    print("CABEÇALHO:")
    print(linhas[0])

    print("PARTIDA:")

    for numero, linha in enumerate(linhas, start=1):

        if time_casa in linha and time_fora in linha:
            print("Linha:", numero)
            print(linha)

procurar_partida(
    "2022_23",
    "N1.csv",
    "Utrecht",
    "Feyenoord"
)

procurar_partida(
    "2023_24",
    "SC2.csv",
    "Edinburgh City",
    "Falkirk"
)

procurar_partida(
    "2024_25",
    "EC.csv",
    "Altrincham",
    "Braintree Town"
)

procurar_partida(
    "2024_25",
    "G1.csv",
    "Lamia",
    "Atromitos"
)

procurar_partida(
    "2009_10",
    "SP2.csv",
    "Gimnastic",
    "Elche"
)