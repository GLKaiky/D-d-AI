import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Carregar o dataset
magias_df = pd.read_csv("Magias.csv", encoding="latin1", delimiter=";", keep_default_na=False, na_values=[])

# Convertendo a coluna 'Descricao' para string, se necessário
magias_df["Descricao"] = magias_df["Descricao"].astype(str)

# Agrupar magias por nível e classe e manter a ordem original das magias
magias_por_nivel_classe = magias_df.groupby(["Nivel", "Classe"]).agg({
    "Nome": list,
    "Descricao": list
}).reset_index()

# Mapear os nomes e descrições das magias para seus índices correspondentes
nome_para_indice = {nome: i for i, nome in enumerate(magias_por_nivel_classe["Nome"].sum())}
descricao_para_indice = {descricao: i for i, descricao in enumerate(magias_por_nivel_classe["Descricao"].sum())}

# Preparar os dados para MultiLabel Binarizer
def binarizar_magias(magias):
    binarizadas = []
    for magia in magias:
        indices_nome = [nome_para_indice[nome] for nome in magia[0]]
        indices_descricao = [descricao_para_indice[descricao] for descricao in magia[1]]
        binarizada = [0] * (len(nome_para_indice) + len(descricao_para_indice))
        for indice in indices_nome + indices_descricao:
            binarizada[indice] = 1
        binarizadas.append(binarizada)
    return binarizadas

y_binarizado = binarizar_magias(magias_por_nivel_classe[["Nome", "Descricao"]].values)

# Definir as features
X = magias_por_nivel_classe[["Nivel", "Classe"]]

# Converter a coluna Classe para valores numéricos
X = pd.get_dummies(X, columns=["Classe"])

# Salvar as colunas de features para uso posterior
feature_columns = X.columns.tolist()

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_binarizado, test_size=0.2, random_state=42)

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

X_train["Nivel"] = pd.to_numeric(X_train["Nivel"], errors='coerce')
X_test["Nivel"] = pd.to_numeric(X_test["Nivel"], errors='coerce')

# Tratar os NaNs resultantes da conversão
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)


# Treinar um modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões para o conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

os.makedirs("Models", exist_ok=True)


# Salvar o modelo treinado e os mapeamentos de nome e descrição
joblib.dump(model, "Models/model_random_forest.pkl")
joblib.dump(nome_para_indice, "Models/nome_para_indice.pkl")
joblib.dump(descricao_para_indice, "Models/descricao_para_indice.pkl")
joblib.dump(feature_columns, "Models/feature_columns.pkl")
