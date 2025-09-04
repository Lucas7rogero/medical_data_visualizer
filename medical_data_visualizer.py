import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("medical_examination.csv")

# cria coluna "overweight"
df["overweight"] = (df["weight"] / (df["height"]/100)**2).apply(lambda x: 1 if x > 25 else 0)

# normaliza "cholesterol" e "gluc"
df["cholesterol"] = df["cholesterol"].apply(lambda x: 0 if x == 1 else 1)
df["gluc"] = df["gluc"].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # organiza os dados no formato longo
    df_cat = pd.melt(
        df, 
        id_vars=["cardio"], 
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # conta ocorrencias por categoria
    df_cat["total"] = 1
    df_cat = df_cat.groupby(["cardio", "variable", "value"], as_index=False).count()
    
    # gera o grafico categorico
    fig = sns.catplot(
        x="variable", y="total", data=df_cat, hue="value", col="cardio", kind="bar"
    ).fig

    fig.savefig("catplot.png")
    return fig


def draw_heat_map():
    # remove dados invalidos e outliers
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # matriz de correlacao
    corr = df_heat.corr()

    # mascara para metade superior
    mask = np.triu(corr)

    fig, ax = plt.subplots(figsize=(12, 12))

    # mapa de calor
    sns.heatmap(
        corr, linewidths=1, annot=True, fmt=".1f", mask=mask, 
        square=True, cbar_kws={"shrink": 0.5}, ax=ax
    )

    fig.savefig("heatmap.png")
    return fig
