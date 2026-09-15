import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# 1. Carregar os dados
df = pd.read_excel("data/dados_frutas.xlsx")

# 2. Definir o alvo e as características
y = df["Fruta"]
caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

# 3. Criar e treinar o modelo Random Forest (100 árvores)
floresta = RandomForestClassifier(n_estimators=100, random_state=42)
floresta.fit(X, y)

# 4. Avaliar o modelo nos dados de treino
previsoes = floresta.predict(X)
acertos = (y == previsoes).sum()
total = len(y)

print("--- RESULTADOS DO RANDOM FOREST ---")
print(f"Total de frutas no dataset: {total}")
print(f"Acertos no treinamento: {acertos} / {total} ({(acertos/total)*100:.2f}%)\n")

# 5. Previsão para novas frutas usando DataFrame (evita warnings)
# Criando um teste com a combinação [1, 1, 1, 1]
nova_fruta = pd.DataFrame([[1, 1, 1, 0]], columns=caracteristicas)
resultado = floresta.predict(nova_fruta)

print("--- PREVISÃO ---")
print(f"Previsão para [1, 1, 1, 1]: {resultado[0]}\n")

# 6. Exibir a Importância das Características
print("--- IMPORTÂNCIA DAS CARACTERÍSTICAS ---")
importancia = floresta.feature_importances_
for nome, valor in zip(caracteristicas, importancia):
    print(f"Característica '{nome}': {valor * 100:.2f}%")