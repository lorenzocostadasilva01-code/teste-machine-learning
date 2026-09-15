import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# 1. Carregar os dados
df = pd.read_excel("data/dados_frutas.xlsx")

# 2. Definir o alvo e as características
y = df["Fruta"]
caracteristicas = ["Arredondada", "Suculenta", "Doce", "Vermelha"]
X = df[caracteristicas]

df.info()

# 4. Criar e treinar o modelo
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# 5. Fazer a previsão
# Exemplo: Arredondada(0), Suculenta(0), Doce(0), Vermelha(0)
nova_fruta = [[0, 0, 0, 0]]
resultado = arvore.predict(nova_fruta)

print("A fruta prevista é:", resultado[0])

# 6. Visualizar a árvore de decisão
plt.figure(figsize=(12, 8))
tree.plot_tree(
    arvore, 
    feature_names=caracteristicas, 
    class_names=arvore.classes_, 
    filled=True, 
    rounded=True
)
plt.show()