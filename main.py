"""Ponto de entrada do projeto.

Exemplos (na raiz do repositório):
    python main.py --listar
    python main.py preprocessar
    python main.py imputacao
    python main.py tudo
"""
import argparse
import runpy
import sys
import time

ETAPAS = {
    "preprocessar": [
        "src.preprocessamento.01_concatenar",
        "src.preprocessamento.02_validar_dados",
        "src.preprocessamento.03_analisar_ausentes",
    ],
    # comparação dos métodos de imputação com validação cruzada (k-fold)
    "imputacao": [
        "src.imputacao.experimento_mice",
        "src.imputacao.experimento_knn",
        "src.imputacao.experimento_missforest",
        "src.imputacao.experimento_gain",
        "src.imputacao.experimento_mediana",
        "src.imputacao.comparar_resultados",
        "src.imputacao.testes_estatisticos",
        "src.imputacao.gerar_graficos",
        "src.imputacao.imputar_base",
    ],
    # odds e engenharia de atributos
    "atributos": [
        "src.atributos.separar_odds",
        "src.atributos.probabilidades",
        "src.atributos.engenharia",
        "src.atributos.matriz",
        "src.atributos.base_completa",
    ],
    # detecção de anomalias — os módulos entram aqui conforme forem criados
    "deteccao": [],
}

TUDO = ["preprocessar", "imputacao", "atributos", "deteccao"]


def rodar(modulo):
    print(f"\n{'=' * 60}\n▶ {modulo}\n{'=' * 60}")
    inicio = time.time()
    runpy.run_module(modulo, run_name="__main__", alter_sys=True)
    print(f"✔ {modulo} ({time.time() - inicio:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline do TP2 — anomalias no futebol")
    parser.add_argument("etapas", nargs="*", choices=[*ETAPAS, "tudo"],
                        help="etapas a executar, na ordem dada")
    parser.add_argument("--listar", action="store_true", help="mostra as etapas e os scripts")
    args = parser.parse_args()

    if args.listar or not args.etapas:
        for nome, modulos in ETAPAS.items():
            print(f"{nome}:")
            for m in modulos or ["(ainda vazio)"]:
                print(f"  - {m}")
        if not args.etapas:
            print("\nUso: python main.py <etapa> [<etapa> ...]   |   python main.py tudo")
        return

    etapas = []
    for e in args.etapas:
        etapas += TUDO if e == "tudo" else [e]

    for etapa in etapas:
        if not ETAPAS[etapa]:
            print(f"\n(etapa '{etapa}' ainda não tem scripts)")
        for modulo in ETAPAS[etapa]:
            rodar(modulo)


if __name__ == "__main__":
    sys.exit(main())
