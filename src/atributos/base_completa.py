"""Base completa: cada partida com estatísticas, odds, probabilidades, todos os
atributos criados e os atributos normalizados da matriz de detecção, em blocos
organizados. Gera também um dicionário de dados com a descrição de cada coluna.

Entradas: data/processed/futebol_atributos.csv
          data/processed/futebol_matriz.csv
Saídas:   data/processed/futebol_base_completa.csv
          results/atributos/dicionario_base_completa.csv
"""
import pandas as pd

from src.config import DADOS_PROCESSADOS, RESULTADOS

CHAVE = ["Div", "Season", "Date", "HomeTeam", "AwayTeam"]

# bloco -> {coluna: descrição}, na ordem em que aparecem na base
DICIONARIO = {
    "1. Identificação": {
        "Div": "Código da liga (ex.: E0 = Premier League, SP1 = La Liga)",
        "Season": "Temporada (ex.: 2015_16)",
        "Date": "Data da partida",
        "HomeTeam": "Mandante",
        "AwayTeam": "Visitante",
    },
    "2. Resultado": {
        "FTHG": "Gols do mandante no jogo",
        "FTAG": "Gols do visitante no jogo",
        "FTR": "Resultado final (H = mandante, D = empate, A = visitante)",
        "HTHG": "Gols do mandante no 1º tempo",
        "HTAG": "Gols do visitante no 1º tempo",
        "HTR": "Resultado no intervalo (H, D, A)",
    },
    "3. Estatísticas do jogo": {
        "HS": "Finalizações do mandante",
        "AS": "Finalizações do visitante",
        "HST": "Finalizações no alvo do mandante",
        "AST": "Finalizações no alvo do visitante",
        "HF": "Faltas cometidas pelo mandante",
        "AF": "Faltas cometidas pelo visitante",
        "HC": "Escanteios do mandante",
        "AC": "Escanteios do visitante",
        "HY": "Amarelos do mandante",
        "AY": "Amarelos do visitante",
        "HR": "Vermelhos do mandante",
        "AR": "Vermelhos do visitante",
        "n_imputados": "Quantas das 14 estatísticas foram preenchidas pelo MICE",
        "imputado": "True se alguma estatística foi imputada",
    },
    "4. Odds (média do mercado)": {
        "odd_casa": "Odd média de vitória do mandante",
        "odd_empate": "Odd média de empate",
        "odd_fora": "Odd média de vitória do visitante",
        "odd_max_casa": "Maior odd do mercado para o mandante",
        "odd_max_empate": "Maior odd do mercado para o empate",
        "odd_max_fora": "Maior odd do mercado para o visitante",
        "odd_mais25": "Odd média de mais de 2,5 gols",
        "odd_menos25": "Odd média de menos de 2,5 gols",
        "odd_fech_casa": "Odd de fechamento do mandante (a partir de 2019/20)",
        "odd_fech_empate": "Odd de fechamento do empate (a partir de 2019/20)",
        "odd_fech_fora": "Odd de fechamento do visitante (a partir de 2019/20)",
        "odd_fech_mais25": "Odd de fechamento de mais de 2,5 gols (a partir de 2019/20)",
        "odd_fech_menos25": "Odd de fechamento de menos de 2,5 gols (a partir de 2019/20)",
    },
    "5. Probabilidades implícitas (sem a margem da casa)": {
        "prob_casa": "Probabilidade de vitória do mandante",
        "prob_empate": "Probabilidade de empate",
        "prob_fora": "Probabilidade de vitória do visitante",
        "margem_1x2": "Margem da casa no mercado de resultado (soma de 1/odd - 1)",
        "prob_mais25": "Probabilidade de mais de 2,5 gols",
        "prob_menos25": "Probabilidade de menos de 2,5 gols",
        "margem_gols": "Margem da casa no mercado de gols",
        "prob_fech_casa": "Probabilidade de fechamento do mandante",
        "prob_fech_empate": "Probabilidade de fechamento do empate",
        "prob_fech_fora": "Probabilidade de fechamento do visitante",
        "margem_fech_1x2": "Margem de fechamento no mercado de resultado",
        "prob_fech_mais25": "Probabilidade de fechamento de mais de 2,5 gols",
        "prob_fech_menos25": "Probabilidade de fechamento de menos de 2,5 gols",
        "margem_fech_gols": "Margem de fechamento no mercado de gols",
    },
    "6. Atributos — mercado x resultado": {
        "prob_resultado": "Probabilidade que o mercado dava ao resultado ocorrido",
        "surpresa_resultado": "-ln(prob_resultado): quanto o resultado contrariou o mercado",
        "surpresa_gols": "-ln(probabilidade do lado de 2,5 gols que ocorreu)",
        "forca_mercado": "prob_casa - prob_fora (> 0: mandante favorito)",
        "gols_esperados": "Gols esperados pelo mercado (Poisson a partir de prob_mais25)",
        "desvio_gols": "Gols reais - gols esperados",
        "dispersao_odds": "Odd máxima / odd média - 1 no resultado ocorrido",
        "mov_casa": "Probabilidade de fechamento - abertura, mandante (2019/20+)",
        "mov_fora": "Probabilidade de fechamento - abertura, visitante (2019/20+)",
        "mov_mais25": "Probabilidade de fechamento - abertura, mais de 2,5 (2019/20+)",
    },
    "7. Atributos — mercado x campo": {
        "dif_finalizacoes": "Finalizações do mandante - do visitante",
        "dif_no_alvo": "Finalizações no alvo do mandante - do visitante",
        "dif_escanteios": "Escanteios do mandante - do visitante",
        "dominio_campo": "Fração das finalizações no alvo feitas pelo mandante (0 a 1)",
        "dominio_esperado": "Domínio esperado pelo mercado: prob_casa / (prob_casa + prob_fora)",
        "desencontro_dominio": "dominio_campo - dominio_esperado",
        "vitoria_dominado": "1 se o vencedor teve menos finalizações no alvo",
        "conversao": "Gols / finalizações no alvo (as duas equipes)",
    },
    "8. Atributos — o próprio jogo": {
        "total_gols": "Gols no jogo",
        "gols_2t": "Gols no 2º tempo",
        "prop_gols_2t": "Fração dos gols no 2º tempo",
        "vantagem_perdida": "1 se quem vencia no intervalo não venceu",
        "virada": "1 se quem vencia no intervalo perdeu",
        "perda_vantagem": "Saldo perdido no 2º tempo por quem vencia no intervalo",
        "total_faltas": "Faltas no jogo",
        "total_cartoes": "Cartões (amarelos + vermelhos) no jogo",
        "total_vermelhos": "Vermelhos no jogo",
        "dif_cartoes": "Cartões do mandante - do visitante",
        "indice_disciplina": "Amarelos + 2,5 x vermelhos",
    },
    "9. Atributos — contexto (antes do jogo)": {
        "forma_casa": "Pontos médios do mandante nos últimos 5 jogos da temporada",
        "forma_fora": "Pontos médios do visitante nos últimos 5 jogos da temporada",
        "dif_forma": "forma_casa - forma_fora",
        "ppj_casa": "Pontos por jogo do mandante na temporada até o jogo",
        "ppj_fora": "Pontos por jogo do visitante na temporada até o jogo",
        "saldo_casa": "Saldo de gols médio do mandante nos últimos 5 jogos",
        "saldo_fora": "Saldo de gols médio do visitante nos últimos 5 jogos",
        "posicao_casa": "Posição do mandante na tabela antes do jogo",
        "posicao_fora": "Posição do visitante na tabela antes do jogo",
        "posicao_rel_casa": "Posição relativa do mandante (0 = líder, 1 = último)",
        "posicao_rel_fora": "Posição relativa do visitante (0 = líder, 1 = último)",
        "dif_posicao": "posicao_casa - posicao_fora (< 0: mandante mais bem colocado)",
        "progresso": "Fração da temporada já disputada (0 a 1)",
        "reta_final": "1 nos últimos 15% da temporada",
    },
    "10. Matriz de detecção (z-score por liga e temporada)": {
        "sem_odds_gols": "1 se a partida não tem odds de 2,5 gols",
        "z_surpresa_resultado": "surpresa_resultado normalizada",
        "z_surpresa_gols": "surpresa_gols normalizada",
        "z_desvio_gols": "desvio_gols normalizado",
        "z_dispersao_odds": "dispersao_odds normalizada (limitada ao percentil 99,9)",
        "z_margem_1x2": "margem_1x2 normalizada",
        "z_desencontro_dominio": "desencontro_dominio normalizado",
        "z_conversao": "conversao normalizada",
        "z_prop_gols_2t": "prop_gols_2t normalizada",
        "z_perda_vantagem": "perda_vantagem normalizada",
        "z_total_faltas": "total_faltas normalizado",
        "z_indice_disciplina": "indice_disciplina normalizado",
        "z_dif_cartoes": "dif_cartoes normalizada",
    },
}

atributos = pd.read_csv(DADOS_PROCESSADOS / "futebol_atributos.csv", low_memory=False)
matriz = pd.read_csv(DADOS_PROCESSADOS / "futebol_matriz.csv", low_memory=False)
print("Atributos:", atributos.shape, "| matriz:", matriz.shape)

extras = matriz[CHAVE + [c for c in matriz.columns if c not in atributos.columns]]
base = atributos.merge(extras, on=CHAVE, how="left", validate="one_to_one")
assert len(base) == len(atributos), "a junção mudou o número de partidas"

ordem = [c for bloco in DICIONARIO.values() for c in bloco]
faltando = [c for c in ordem if c not in base.columns]
sobrando = [c for c in base.columns if c not in ordem]
assert not faltando, f"colunas do dicionário que não existem na base: {faltando}"
assert not sobrando, f"colunas sem descrição no dicionário: {sobrando}"

base = base[ordem].sort_values(["Div", "Season", "Date", "HomeTeam"]).reset_index(drop=True)
base.to_csv(DADOS_PROCESSADOS / "futebol_base_completa.csv", index=False)

dicionario = pd.DataFrame(
    [{"bloco": bloco, "coluna": c, "descricao": d,
      "preenchidos_pct": round(base[c].notna().mean() * 100, 1)}
     for bloco, colunas in DICIONARIO.items() for c, d in colunas.items()]
)
pasta = RESULTADOS / "atributos"
pasta.mkdir(parents=True, exist_ok=True)
dicionario.to_csv(pasta / "dicionario_base_completa.csv", index=False)

print("\nColunas por bloco:")
print(dicionario.groupby("bloco", sort=False).size().to_string())
print(f"\nBase salva em data/processed/futebol_base_completa.csv: {base.shape}")
print("Dicionário salvo em results/atributos/dicionario_base_completa.csv")
