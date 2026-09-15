import pandas as pd
# 1. Troca a importação do tree pelo ensemble
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# 2. Carregar os dados
df = pd.read_excel("data/dados_frutas.xlsx")

# 3. Definir o alvo e as características
y = df["Fruta"]
caracteristicas = ["Arredondada", "Suculenta", "Doce", "Vermelha"]
X = df[caracteristicas]

# 4. Criar e treinar o modelo Random Forest
# n_estimators=100 cria uma floresta com 100 árvores de decisão
floresta = RandomForestClassifier(n_estimators=100, random_state=42)
floresta.fit(X, y)

# 5. Avaliar o modelo
previsoes = floresta.predict(X)
acertos = (y == previsoes).sum()
total = len(y)

print("--- RESULTADOS DO RANDOM FOREST ---")
print(f"Total de frutas no dataset: {total}")
print(f"Acertos no treinamento: {acertos} / {total} ({(acertos/total)*100:.2f}%)\n")

# 6. Fazer uma previsão para uma nova combinação
# Exemplo: Arredondada(1), Suculenta(1), Doce(1), Vermelha(1)
nova_fruta = [[1, 1, 1, 1]]
resultado = floresta.predict(nova_fruta)

print(f"Previsão para [1, 1, 1, 1]: {resultado[0]}")

primeira_arvore = floresta.estimators_[20]

# Desenha a árvore escolhida
plt.figure(figsize=(12, 8))
tree.plot_tree(
    primeira_arvore,
    feature_names=caracteristicas,
    class_names=floresta.classes_,
    filled=True,
    rounded=True
)
plt.title("Primeira Árvore do Random Forest (Índice 0)")
plt.show()