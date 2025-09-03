import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carrega os dados
df = pd.read_csv("medical_examination.csv")

# Cria coluna 'overweight' (IMC > 25 = 1, senão 0)
df["overweight"] = (df["weight"] / (df["height"]/100)**2).apply(lambda x: 1 if x > 25 else 0)

# Normaliza 'cholesterol' e 'gluc' (1 → 0, >1 → 1)
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # Reorganiza os dados no formato longo
    df_cat = pd.melt(
        df, 
        id_vars=["cardio"], 
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # Conta ocorrências por categoria
    df_cat["total"] = 1
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).count()
    
    # Gera o gráfico categórico
    fig = sns.catplot(
        x="variable", y="total", data=df_cat, hue="value", col="cardio", kind="bar"
    ).fig

    fig.savefig("catplot.png")
    return fig


def draw_heat_map():
    # Remove dados inválidos e outliers
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # Matriz de correlação
    corr = df_heat.corr()

    # Máscara para metade superior
    mask = np.triu(corr)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Mapa de calor
    sns.heatmap(
        corr, linewidths=1, annot=True, fmt=".1f", mask=mask, 
        square=True, cbar_kws={"shrink": 0.5}, ax=ax
    )

    fig.savefig("heatmap.png")
    return fig
