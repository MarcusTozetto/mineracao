# Detecção de anomalias em partidas de futebol

Trabalho prático de Mineração de Dados (UFSJ). O objetivo é atribuir a cada
partida um score de anomalia entre 0 e 1 a partir das estatísticas do jogo e
das odds de apostas. O score indica um comportamento fora do padrão, não uma
acusação de manipulação.

O pipeline tem três etapas:

1. **Pré-processamento** — junta os arquivos do Football-Data, valida e limpa a base.
2. **Imputação** — compara MICE, KNN, MissForest e GAIN (com a mediana como
   linha de base), com validação cruzada (k-fold) e teste de Wilcoxon, e aplica
   o vencedor (MICE) à base, gerando `data/processed/futebol_imputado.csv`.
3. **Atributos** — recorta o período com odds (a partir de 2005/06), unifica as
   colunas de odds e constrói os atributos usados na detecção.
4. **Detecção** — score de anomalia por partida.

## Dados

Os arquivos brutos vêm do [Football-Data](https://www.football-data.co.uk/)
e ficam versionados em `data/raw/<temporada>/<liga>.csv`. As bases geradas
(`data/processed/`) não são versionadas e são recriadas pelo pipeline.

## Estrutura

```
main.py                  executa o pipeline por etapas
data/
  raw/                   dados brutos, por temporada
  processed/             bases geradas (não versionadas)
results/
  imputacao/             métricas da comparação de imputação
  deteccao/              scores de anomalia
  figures/               gráficos
src/
  config.py              caminhos do projeto
  preprocessamento/      01_concatenar → 02_validar_dados → 03_analisar_ausentes
  imputacao/             experimentos por método, comparação, gráficos e imputação final
  atributos/             odds, probabilidades e engenharia de atributos
  deteccao/              detecção de anomalias
```

## Como rodar

Na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --listar          # etapas e scripts
python main.py preprocessar      # gera data/processed/futebol_limpo.csv
python main.py imputacao         # comparação dos métodos + base imputada
python main.py atributos         # base com odds e atributos
python main.py tudo              # pipeline completo
```

Cada script também roda sozinho, por exemplo
`python -m src.imputacao.experimento_mice`.
