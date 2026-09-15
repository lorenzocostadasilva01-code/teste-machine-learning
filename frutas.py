import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# 1. Carregar os dados
df = pd.read_excel("data/dados_frutas.xlsx")

# 2. Definir o alvo e as características
y = df["Fruta"]
caracteristicas = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
X = df[caracteristicas]

# 3. Criar e treinar o modelo com a base COMPLETA
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# 4. Avaliar o modelo nos próprios dados de treino
previsoes = arvore.predict(X)
acertos = (y == previsoes).sum()
total = len(y)

print("--- RESULTADOS DO MODELO ---")
print(f"Total de frutas no dataset: {total}")
print(f"Acertos no treinamento: {acertos} / {total} ({(acertos/total)*100:.2f}%)\n")

# 5. Fazer uma previsão para uma fruta inédita/customizada
# Exemplo: Arredondada(1), Suculenta(1), Doce(1), Vermelha(1)
novas_frutas = [
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0]
]
resultado = arvore.predict(novas_frutas)

for i, fruta_prevista in enumerate(resultado, start=1):
    print(f"Fruta {i}: {fruta_prevista}")

# 6. Visualizar a Árvore de Decisão gerada
plt.figure(figsize=(14, 8))
tree.plot_tree(
    arvore, 
    feature_names=caracteristicas, 
    class_names=arvore.classes_, 
    filled=True, 
    rounded=True
)
