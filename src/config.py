"""Caminhos do projeto, centralizados.

Todos os scripts importam daqui, então funcionam independentemente da pasta
de onde são executados.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"

RESULTADOS = RAIZ / "results"
RESULTADOS_IMPUTACAO = RESULTADOS / "imputacao"
RESULTADOS_DETECCAO = RESULTADOS / "deteccao"
FIGURAS = RESULTADOS / "figures"

for _pasta in (DADOS_PROCESSADOS, RESULTADOS_IMPUTACAO, RESULTADOS_DETECCAO, FIGURAS):
    _pasta.mkdir(parents=True, exist_ok=True)
