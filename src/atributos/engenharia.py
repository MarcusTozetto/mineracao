"""Engenharia de atributos.

Grupo 1 — o resultado bateu com o que o mercado esperava?
    prob_resultado      probabilidade que o mercado dava ao resultado ocorrido
    surpresa_resultado  -ln(prob_resultado): zebra de 5% -> 3,0; favorito de 70% -> 0,36
    surpresa_gols       -ln(prob do lado de 2,5 gols que ocorreu)
    forca_mercado       prob_casa - prob_fora (> 0: mandante favorito)
    dispersao_odds      odd máxima / odd média - 1 no resultado ocorrido
                        (casas discordando sobre o resultado que aconteceu)
    mov_casa, mov_fora, mov_mais25
                        probabilidade de fechamento - probabilidade de abertura
                        (só a partir de 2019/20; NaN antes)

    gols_esperados      gols que o mercado esperava: a média λ de uma Poisson
                        com P(total > 2,5) = prob_mais25
    desvio_gols         total_gols - gols_esperados

Grupo 2 — o domínio em campo bateu com o favoritismo?
    dif_finalizacoes, dif_no_alvo, dif_escanteios   casa - fora
    dominio_campo       fração das finalizações no alvo feitas pelo mandante (0 a 1)
    dominio_esperado    prob_casa / (prob_casa + prob_fora): o domínio que o
                        mercado esperava, na mesma escala
    desencontro_dominio dominio_campo - dominio_esperado (> 0: mandante dominou
                        mais do que o mercado previa)
    vitoria_dominado    1 se o vencedor teve menos finalizações no alvo que o perdedor
    conversao           gols / finalizações no alvo (as duas equipes)

Grupo 3 — o próprio jogo
    total_gols, gols_2t     gols no jogo e no 2º tempo
    prop_gols_2t        fração dos gols feita no 2º tempo
    vantagem_perdida    1 se quem vencia no intervalo não venceu o jogo
    virada              1 se quem vencia no intervalo perdeu o jogo
    perda_vantagem      gols de saldo que quem vencia no intervalo perdeu no
                        2º tempo (0 se o intervalo terminou empatado; negativo
                        se ampliou a vantagem): versão contínua da virada
    total_faltas, total_cartoes, total_vermelhos
    dif_cartoes         cartões (amarelo + vermelho) casa - fora
    indice_disciplina   amarelos + 2,5 x vermelhos (as duas equipes): um
                        índice único, para a disciplina não pesar em dobro

Grupo 4 — contexto dos times (só jogos anteriores, sem vazar o resultado)
    forma_casa, forma_fora  média de pontos nos últimos 5 jogos da temporada
    dif_forma           forma_casa - forma_fora
    ppj_casa, ppj_fora  pontos por jogo acumulados na temporada até o jogo
    saldo_casa, saldo_fora  saldo de gols por jogo nos últimos 5 jogos
    posicao_casa, posicao_fora  posição na tabela da liga ANTES do jogo
                        (pontos, depois saldo, depois gols marcados)
    posicao_rel_casa, posicao_rel_fora  posição / nº de times (0 = líder, 1 = último)
    dif_posicao         posicao_casa - posicao_fora (< 0: mandante mais bem colocado)
    progresso           fração da temporada já disputada (0 a 1)
    reta_final          1 nos últimos 15% da temporada
    Nos primeiros jogos de cada time na temporada (menos de 3 anteriores) a
    forma e a posição ficam NaN: a tabela ainda não diz nada.

Entrada: data/processed/futebol_probabilidades.csv
Saída:   data/processed/futebol_atributos.csv
"""
import numpy as np
import pandas as pd

from src.config import DADOS_PROCESSADOS

CASAS_DECIMAIS = 4

df = pd.read_csv(DADOS_PROCESSADOS / "futebol_probabilidades.csv", low_memory=False)
print("Base com probabilidades:", df.shape)

resultado = df["FTR"]
casa, empate, fora = resultado.eq("H"), resultado.eq("D"), resultado.eq("A")
total_gols = df["FTHG"] + df["FTAG"]
mais25 = total_gols > 2.5

# ------------------------------------------------------------------
# Grupo 1 — mercado x resultado
# ------------------------------------------------------------------
df["prob_resultado"] = np.select(
    [casa, empate, fora], [df["prob_casa"], df["prob_empate"], df["prob_fora"]],
    default=np.nan,
)
df["surpresa_resultado"] = -np.log(df["prob_resultado"])

prob_gols = np.where(mais25, df["prob_mais25"], df["prob_menos25"])
df["surpresa_gols"] = -np.log(prob_gols)

df["forca_mercado"] = df["prob_casa"] - df["prob_fora"]


def gols_esperados_poisson(prob_mais25, iteracoes=60):
    """λ tal que P(Poisson(λ) > 2,5) = prob_mais25, por bisseção (vetorizada).

    P(X > 2,5) = 1 - e^(-λ) (1 + λ + λ²/2) cresce com λ, então a bisseção
    converge; 60 iterações dão precisão muito acima de 4 casas decimais.
    """
    p = np.asarray(prob_mais25, dtype=float)
    baixo, alto = np.full_like(p, 0.01), np.full_like(p, 10.0)
    for _ in range(iteracoes):
        meio = (baixo + alto) / 2
        p_meio = 1 - np.exp(-meio) * (1 + meio + meio ** 2 / 2)
        baixo = np.where(p_meio < p, meio, baixo)
        alto = np.where(p_meio < p, alto, meio)
    return np.where(np.isnan(p), np.nan, (baixo + alto) / 2)


df["gols_esperados"] = gols_esperados_poisson(df["prob_mais25"])
df["desvio_gols"] = total_gols - df["gols_esperados"]

odd_media = np.select([casa, empate, fora],
                      [df["odd_casa"], df["odd_empate"], df["odd_fora"]], default=np.nan)
odd_max = np.select([casa, empate, fora],
                    [df["odd_max_casa"], df["odd_max_empate"], df["odd_max_fora"]],
                    default=np.nan)
df["dispersao_odds"] = odd_max / odd_media - 1
# odd máxima abaixo da média é impossível: erro na fonte
erros = df["dispersao_odds"] < 0
df.loc[erros, "dispersao_odds"] = np.nan
print(f"dispersao_odds: {erros.sum()} partidas com odd máxima < média (erro na fonte) -> NaN")

df["mov_casa"] = df["prob_fech_casa"] - df["prob_casa"]
df["mov_fora"] = df["prob_fech_fora"] - df["prob_fora"]
df["mov_mais25"] = df["prob_fech_mais25"] - df["prob_mais25"]

# ------------------------------------------------------------------
# Grupo 2 — mercado x campo
# ------------------------------------------------------------------
df["dif_finalizacoes"] = df["HS"] - df["AS"]
df["dif_no_alvo"] = df["HST"] - df["AST"]
df["dif_escanteios"] = df["HC"] - df["AC"]

no_alvo = df["HST"] + df["AST"]
# sem nenhuma finalização no alvo: domínio neutro
df["dominio_campo"] = np.where(no_alvo > 0, df["HST"] / no_alvo.replace(0, np.nan), 0.5)
df["dominio_esperado"] = df["prob_casa"] / (df["prob_casa"] + df["prob_fora"])
df["desencontro_dominio"] = df["dominio_campo"] - df["dominio_esperado"]

df["vitoria_dominado"] = ((casa & (df["HST"] < df["AST"]))
                          | (fora & (df["AST"] < df["HST"]))).astype(int)

df["conversao"] = total_gols / no_alvo.replace(0, np.nan)

# ------------------------------------------------------------------
# Grupo 3 — o próprio jogo
# ------------------------------------------------------------------
df["total_gols"] = total_gols
df["gols_2t"] = total_gols - (df["HTHG"] + df["HTAG"])
df["prop_gols_2t"] = df["gols_2t"] / total_gols.replace(0, np.nan)
# vantagem perdida: quem vencia no intervalo não venceu (empate ou derrota)
df["vantagem_perdida"] = (df["HTR"].isin(["H", "A"]) & (df["HTR"] != df["FTR"])).astype(int)
# virada: quem vencia no intervalo perdeu o jogo
df["virada"] = (((df["HTR"] == "H") & (df["FTR"] == "A"))
                | ((df["HTR"] == "A") & (df["FTR"] == "H"))).astype(int)
saldo_intervalo = df["HTHG"] - df["HTAG"]
saldo_final = df["FTHG"] - df["FTAG"]
df["perda_vantagem"] = -np.sign(saldo_intervalo) * (saldo_final - saldo_intervalo)
df["total_faltas"] = df["HF"] + df["AF"]
df["total_cartoes"] = df["HY"] + df["AY"] + df["HR"] + df["AR"]
df["total_vermelhos"] = df["HR"] + df["AR"]
df["dif_cartoes"] = (df["HY"] + df["HR"]) - (df["AY"] + df["AR"])
df["indice_disciplina"] = (df["HY"] + df["AY"]) + 2.5 * (df["HR"] + df["AR"])

# ------------------------------------------------------------------
# Grupo 4 — contexto dos times
# ------------------------------------------------------------------
JANELA, MINIMO = 5, 3


def posicoes_na_tabela(df):
    """Posição de cada time na tabela imediatamente ANTES de cada jogo.

    Para cada liga e temporada, acumula pontos, saldo e gols de cada time data a
    data e ranqueia os times com o estado do dia anterior (jogos do mesmo dia
    não contam). Critério: pontos, depois saldo de gols, depois gols marcados.
    Times empatados em tudo recebem a mesma posição (a melhor).

    Retorna um DataFrame indexado por id_jogo com posicao_casa, posicao_fora e
    n_times.
    """
    saida = []
    for (_, _), jogos in df.groupby(["Div", "Season"], sort=False):
        pontos_c = np.select([jogos["FTR"].eq("H"), jogos["FTR"].eq("D")], [3, 1], 0)
        pontos_f = np.select([jogos["FTR"].eq("A"), jogos["FTR"].eq("D")], [3, 1], 0)
        longo = pd.concat([
            pd.DataFrame({"Date": jogos["Date"], "time": jogos["HomeTeam"],
                          "pontos": pontos_c, "saldo": jogos["FTHG"] - jogos["FTAG"],
                          "gols": jogos["FTHG"]}),
            pd.DataFrame({"Date": jogos["Date"], "time": jogos["AwayTeam"],
                          "pontos": pontos_f, "saldo": jogos["FTAG"] - jogos["FTHG"],
                          "gols": jogos["FTAG"]}),
        ])
        # totais acumulados ao FIM de cada data, uma coluna por time
        por_dia = longo.groupby(["Date", "time"])[["pontos", "saldo", "gols"]].sum()
        acumulado = {c: por_dia[c].unstack(fill_value=0).sort_index().cumsum()
                     for c in ["pontos", "saldo", "gols"]}
        # estado ANTES de cada data = acumulado da data anterior
        antes = {c: t.shift(1, fill_value=0) for c, t in acumulado.items()}
        # chave única de ordenação: pontos > saldo > gols
        chave = antes["pontos"] * 1e6 + (antes["saldo"] + 1000) * 1e3 + antes["gols"]
        posicao = chave.rank(axis=1, ascending=False, method="min")

        posicao = posicao.stack()  # índice (data, time)
        datas = jogos["Date"].to_numpy()
        saida.append(pd.DataFrame({
            "id_jogo": jogos["id_jogo"].to_numpy(),
            "posicao_casa": posicao.reindex(list(zip(datas, jogos["HomeTeam"]))).to_numpy(),
            "posicao_fora": posicao.reindex(list(zip(datas, jogos["AwayTeam"]))).to_numpy(),
            "n_times": acumulado["pontos"].shape[1],
        }))
    return pd.concat(saida).set_index("id_jogo")


df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Div", "Season", "Date", "HomeTeam"]).reset_index(drop=True)
df["id_jogo"] = np.arange(len(df))

# recalcula depois de reordenar a base
casa, empate, fora = df["FTR"].eq("H"), df["FTR"].eq("D"), df["FTR"].eq("A")
pontos_casa = np.select([casa, empate], [3, 1], 0)
pontos_fora = np.select([fora, empate], [3, 1], 0)

# formato longo: uma linha por time por jogo
jogos_time = pd.concat([
    pd.DataFrame({"id_jogo": df["id_jogo"], "Div": df["Div"], "Season": df["Season"],
                  "Date": df["Date"], "time": df["HomeTeam"], "lado": "casa",
                  "pontos": pontos_casa, "saldo": df["FTHG"] - df["FTAG"]}),
    pd.DataFrame({"id_jogo": df["id_jogo"], "Div": df["Div"], "Season": df["Season"],
                  "Date": df["Date"], "time": df["AwayTeam"], "lado": "fora",
                  "pontos": pontos_fora, "saldo": df["FTAG"] - df["FTHG"]}),
]).sort_values(["Div", "Season", "time", "Date", "id_jogo"])

g = jogos_time.groupby(["Div", "Season", "time"], sort=False)
# shift(1): só jogos anteriores entram na conta
anteriores_pontos = g["pontos"].shift(1)
anteriores_saldo = g["saldo"].shift(1)
chave = [jogos_time["Div"], jogos_time["Season"], jogos_time["time"]]
jogos_time["forma"] = anteriores_pontos.groupby(chave).transform(
    lambda x: x.rolling(JANELA, min_periods=MINIMO).mean())
jogos_time["saldo_recente"] = anteriores_saldo.groupby(chave).transform(
    lambda x: x.rolling(JANELA, min_periods=MINIMO).mean())
jogos_time["ppj"] = anteriores_pontos.groupby(chave).transform(
    lambda x: x.expanding(min_periods=MINIMO).mean())
jogos_time["n_jogo"] = g.cumcount() + 1

# jogos que um time costuma ter na temporada daquela liga (temporadas completas)
por_temporada = jogos_time.groupby(["Div", "Season", "time"]).size() \
    .groupby(["Div", "Season"]).max()
esperado = por_temporada.groupby("Div").median().rename("jogos_esperados")
jogos_time = jogos_time.join(esperado, on="Div")
jogos_time["progresso"] = (jogos_time["n_jogo"] / jogos_time["jogos_esperados"]).clip(upper=1)

for lado in ["casa", "fora"]:
    parte = jogos_time[jogos_time["lado"] == lado].set_index("id_jogo")
    df[f"forma_{lado}"] = df["id_jogo"].map(parte["forma"])
    df[f"ppj_{lado}"] = df["id_jogo"].map(parte["ppj"])
    df[f"saldo_{lado}"] = df["id_jogo"].map(parte["saldo_recente"])
    df[f"progresso_{lado}"] = df["id_jogo"].map(parte["progresso"])

df["dif_forma"] = df["forma_casa"] - df["forma_fora"]

tabela = posicoes_na_tabela(df)
jogos_anteriores = {lado: jogos_time[jogos_time["lado"] == lado].set_index("id_jogo")["n_jogo"] - 1
                    for lado in ["casa", "fora"]}
for lado in ["casa", "fora"]:
    posicao = df["id_jogo"].map(tabela[f"posicao_{lado}"])
    # com menos de MINIMO jogos a tabela ainda não diz nada
    posicao = posicao.where(df["id_jogo"].map(jogos_anteriores[lado]) >= MINIMO)
    df[f"posicao_{lado}"] = posicao
    df[f"posicao_rel_{lado}"] = (posicao - 1) / (df["id_jogo"].map(tabela["n_times"]) - 1)
df["dif_posicao"] = df["posicao_casa"] - df["posicao_fora"]
df["progresso"] = df[["progresso_casa", "progresso_fora"]].mean(axis=1)
df["reta_final"] = (df["progresso"] >= 0.85).astype(int)
df = df.drop(columns=["progresso_casa", "progresso_fora", "id_jogo"])

ATRIBUTOS = [
    "prob_resultado", "surpresa_resultado", "surpresa_gols", "forca_mercado",
    "gols_esperados", "desvio_gols",
    "dispersao_odds", "mov_casa", "mov_fora", "mov_mais25",
    "dif_finalizacoes", "dif_no_alvo", "dif_escanteios",
    "dominio_campo", "dominio_esperado", "desencontro_dominio",
    "vitoria_dominado", "conversao",
    "total_gols", "gols_2t", "prop_gols_2t", "vantagem_perdida", "virada", "perda_vantagem",
    "total_faltas", "total_cartoes", "total_vermelhos", "dif_cartoes",
    "indice_disciplina",
    "forma_casa", "forma_fora", "dif_forma", "ppj_casa", "ppj_fora",
    "saldo_casa", "saldo_fora", "posicao_casa", "posicao_fora",
    "posicao_rel_casa", "posicao_rel_fora", "dif_posicao", "progresso", "reta_final",
]
df[ATRIBUTOS] = df[ATRIBUTOS].round(CASAS_DECIMAIS)

# ------------------------------------------------------------------
# Conferências
# ------------------------------------------------------------------
print("\nResumo dos atributos:")
print(df[ATRIBUTOS].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())

print(f"\nVitórias de quem foi dominado em campo: {df['vitoria_dominado'].mean():.1%}")
print("Correlação forca_mercado x dif_no_alvo:",
      round(df["forca_mercado"].corr(df["dif_no_alvo"]), 3))
print(f"Gols esperados (média) {df['gols_esperados'].mean():.2f} x gols reais {total_gols.mean():.2f}")
print("Correlação forca_mercado x dif_forma:",
      round(df["forca_mercado"].corr(df["dif_forma"]), 3))
print(f"Vantagem perdida: {df['vantagem_perdida'].mean():.1%} | viradas: {df['virada'].mean():.1%} | jogos na reta final: {df['reta_final'].mean():.1%}")

colunas = ["Div", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
           "prob_resultado", "surpresa_resultado"]
print("\nMaiores surpresas no resultado:")
print(df.nlargest(5, "surpresa_resultado")[colunas].to_string(index=False))

df.to_csv(DADOS_PROCESSADOS / "futebol_atributos.csv", index=False)
print(f"\nBase salva em data/processed/futebol_atributos.csv: {df.shape}")
