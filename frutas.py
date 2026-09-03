import pandas as pd
from sklearn import tree

# 1. Carregar os dados
df = pd.read_excel("data/dados_frutas.xlsx")

# 2. Definir o alvo e as características
y = df["Fruta"]
caracteristicas = ["Arredondada", "Suculenta", "Doce", "Vermelha"]
X = df[caracteristicas]

# 4. Criar e treinar o modelo
# random_state é usado para garantir que os resultados sejam reproduzíveis (tipo uma seed do minecraft)
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# 5. Fazer a previsão
# Exemplo: Arredondada(1), Suculenta(0), Doce(1), Vermelha(1)
nova_fruta = [[0, 0, 0, 0]]
resultado = arvore.predict(nova_fruta)

print("A fruta prevista é:", resultado[0])