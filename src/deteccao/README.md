# src/deteccao — detecção de anomalias

Etapa final do pipeline: dar a cada partida um **score de anomalia entre 0 e 1**
com Isolation Forest, LOF e One-Class SVM. Score alto = partida fora do padrão
(não é acusação de manipulação).

## Como chegar até aqui

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py preprocessar imputacao atributos   # gera a matriz (ou só "atributos",
                                                 # se data/processed já tiver a base imputada)
```

## Entrada: `data/processed/futebol_matriz.csv`

107.604 partidas (2005/06 a 2024/25, 22 ligas), **sem valores vazios**.

| Colunas | Uso |
|---|---|
| `Div`, `Season`, `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR` | identificação — **não entram no modelo** |
| `imputado`, `sem_odds_gols` | controle — **não entram no modelo** (servem para conferir se o detector não está apontando partidas só por terem dado imputado) |
| `z_*` (12 colunas) | **os atributos do modelo** |

Os 12 atributos já estão normalizados (z-score dentro de cada liga e temporada:
média 0, desvio 1). Todos medem **desvio em relação ao esperado**:

| Atributo | O que mede |
|---|---|
| `z_surpresa_resultado` | o resultado contrariou as odds? (−ln da probabilidade do resultado ocorrido) |
| `z_surpresa_gols` | o mesmo para o mercado de mais/menos de 2,5 gols |
| `z_desvio_gols` | gols reais − gols esperados pelo mercado |
| `z_dispersao_odds` | o quanto as casas de apostas discordavam sobre o resultado ocorrido |
| `z_margem_1x2` | margem da casa (incerteza do mercado) |
| `z_desencontro_dominio` | domínio em campo (finalizações no alvo) − domínio esperado pelas odds |
| `z_conversao` | gols / finalizações no alvo |
| `z_prop_gols_2t` | fração dos gols no 2º tempo |
| `z_perda_vantagem` | saldo perdido no 2º tempo por quem vencia no intervalo |
| `z_total_faltas` | faltas |
| `z_indice_disciplina` | amarelos + 2,5 × vermelhos |
| `z_dif_cartoes` | desequilíbrio de cartões entre as equipes |

```python
import pandas as pd
from src.config import DADOS_PROCESSADOS

matriz = pd.read_csv(DADOS_PROCESSADOS / "futebol_matriz.csv")
X = matriz.filter(like="z_")          # 107604 x 12
```

Todos os demais atributos (probabilidades, favoritismo, forma dos times, reta
final, odds de fechamento…) estão em `data/processed/futebol_atributos.csv`,
com as mesmas colunas de identificação, para **interpretar** as partidas
apontadas.

## Decisões já tomadas — por favor mantenha

- **Não colocar atributos de contexto no modelo** (probabilidades, `forca_mercado`,
  forma, `reta_final`, diferença de finalizações). O detector aponta o que é raro;
  um jogo entre um gigante e um pequeno é raro mas não é estranho. Testamos: com
  eles, o topo do ranking virou "jogos desequilibrados". Use-os só para explicar
  e analisar os resultados.
- **Não usar atributos binários (0/1)**: o Isolation Forest isola um evento raro
  de 0/1 em poucos cortes e ele domina o ranking (aconteceu com "virada").
- **Não reescalar a matriz** — já está em z-score por liga/temporada. LOF e
  One-Class SVM dependem da escala; o Isolation Forest não.

## O que falta fazer

1. **Três detectores**, cada um num script desta pasta:
   - `isolation_forest.py` — ex.: `IsolationForest(n_estimators=300, random_state=42)`
   - `lof.py` — `LocalOutlierFactor(novelty=False)`; com 107 mil linhas, testar
     `n_neighbors` (ex.: 20, 50) e usar `n_jobs=-1`
   - `ocsvm.py` — `OneClassSVM` com kernel RBF é O(n²): treinar numa amostra
     (ex.: 10–20 mil partidas) ou usar `SGDOneClassSVM` com `Nystroem`
   - Fixar `random_state=42` em tudo.
2. **Score de 0 a 1 por método**: converter o score bruto em percentil
   (`rank(pct=True)`), para os três ficarem na mesma escala.
3. **Score combinado**: média dos três percentis (e reportar a concordância
   entre os métodos — ex.: correlação de Spearman e sobreposição do top 1%).
4. **Saída**: `data/processed/futebol_scores.csv` com as colunas de identificação
   + `score_if`, `score_lof`, `score_ocsvm`, `score_final`; e um resumo em
   `results/deteccao/` (top 100 partidas com os atributos que mais pesaram).
5. **Registrar os scripts** na lista `"deteccao"` do `main.py`
   (`python main.py deteccao` deve rodar tudo).
6. **Conferências mínimas**:
   - o top não deve ter proporção de `imputado` muito maior que a base (~2%);
   - distribuição do top por liga e temporada (nenhuma liga deve dominar sem motivo);
   - quais atributos têm |z| maior no top 200.

## Referência do teste feito com Isolation Forest (12 atributos)

No top 200: surpresa do resultado média 2,06 (base 1,01); atributos que mais
pesam: `z_dispersao_odds`, `z_surpresa_resultado`, `z_surpresa_gols`,
`z_desencontro_dominio`. Exemplos no topo: Man City 0×0 Huddersfield (2017/18,
City favorito com 88%), Hamburg 1×5 Sandhausen (2019/20).

## Depois da detecção (validação, sem rótulos)

Casos de manipulação documentados, anomalias sintéticas injetadas e busca de
notícias para as partidas de maior score. A base recebe o rótulo
2 = manipulação confirmada, 1 = suspeita documentada, NaN = sem informação.
