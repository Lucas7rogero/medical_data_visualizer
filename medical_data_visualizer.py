import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importa os dados
df = pd.read_csv("medical_examination.csv")

# Adiciona a coluna 'overweight' (sobrepeso)
# Calcula o IMC (peso em kg / (altura em metros)^2)
# Se o IMC for > 25, define 'overweight' como 1 (sim), caso contrário, 0 (não)
df["overweight"] = (df["weight"] / (df["height"]/100)**2).apply(lambda x: 1 if x > 25 else 0)

# Normaliza os dados, onde 0 é sempre 'bom' e 1 é sempre 'ruim'.
# Se o valor de 'cholesterol' ou 'gluc' for 1, torna o valor 0. Se o valor for maior que 1, torna o valor 1.
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)

# Desenha o Gráfico Categórico
def draw_cat_plot():
    # Cria um DataFrame para o gráfico categórico usando `pd.melt`,
    # utilizando apenas os valores de 'cholesterol', 'gluc', 'smoke', 'alco', 'active' e 'overweight'.
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    # Agrupa e reformata os dados para dividi-los por 'cardio'.
    # Mostra a contagem de cada característica. É necessário renomear uma das colunas
    # para que o catplot funcione corretamente.
    df_cat["total"] = 1 # Adiciona uma coluna 'total' para contagem
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).count()
    
    # Desenha o catplot com 'sns.catplot()'
    fig = sns.catplot(x="variable", y="total", data=df_cat, hue="value", col="cardio", kind="bar").fig

    # Não modifique as próximas duas linhas
    fig.savefig("catplot.png")
    return fig


# Desenha o Mapa de Calor
def draw_heat_map():
    # Limpa os dados
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) & # Pressão diastólica não é maior que a sistólica
        (df["height"] >= df["height"].quantile(0.025)) & # Altura entre o percentil 2.5 e 97.5
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) & # Peso entre o percentil 2.5 e 97.5
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Calcula a matriz de correlação
    corr = df_heat.corr()

    # Gera uma máscara para o triângulo superior (redundante)
    mask = np.triu(corr)

    # Configura a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # Desenha o mapa de calor com 'sns.heatmap()'
    sns.heatmap(corr, linewidths=1, annot=True, fmt=".1f", mask=mask, square=True, cbar_kws={"shrink": 0.5}, ax=ax)

    # Não modifique as próximas duas linhas
    fig.savefig("heatmap.png")
    return fig