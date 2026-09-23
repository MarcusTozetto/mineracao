import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#string de busca "water loss prediction" OR "water leakage detection" AND "machine learning" AND "time series"
base_original = pd.read_csv(r"/home/marcus-tozetto/Área de trabalho/mineracao-futebol/data/processed/futebol_consolidado.csv")
for variavel in base_original.columns:
    print(variavel)

base_original_sp = base_original[base_original["sigla_uf"] == "SP"]
agregando_ano = base_original_sp.groupby("ano")["indice_perda_distribuicao_agua"].mean().reset_index()

plt.figure()
sns.barplot(data = agregando_ano,
            x = "ano",
            y = "indice_perda_distribuicao_agua"
            )

plt.xticks(rotation = 90)

plt.show()