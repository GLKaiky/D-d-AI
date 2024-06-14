import pandas as pd
import joblib
import os

def carregar_modelo_e_mapeamentos(model_path):
    # Carregar o modelo treinado
    model = joblib.load(os.path.join(model_path, "model_random_forest.pkl"))
    
    # Carregar os mapeamentos de nome e descrição para índices
    nome_para_indice = joblib.load(os.path.join(model_path, "nome_para_indice.pkl"))
    descricao_para_indice = joblib.load(os.path.join(model_path,"descricao_para_indice.pkl"))
    
    return model, nome_para_indice, descricao_para_indice

def obter_magias_por_nivel_e_classe(nivel, classe, feature_columns, model_path):
    # Carregat o modelo e os mapeamentos
    model, nome_para_indice, descricao_para_indice = carregar_modelo_e_mapeamentos(model_path)
    
    # Preparar os dados para previsão
    data = pd.DataFrame({"Nivel": [nivel], "Classe": [classe]})
    data = pd.get_dummies(data, columns=["Classe"])
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_columns]
    
    # Fazer a previsão
    magias_pred = model.predict(data)
    
    # Converter as previsões em nomes e descrições
    nomes_pred = [nome for nome, indice in nome_para_indice.items() if magias_pred[0][indice] == 1]
    descricoes_pred = [descricao for descricao, indice in descricao_para_indice.items() if magias_pred[0][indice] == 1]
    
    # Criar uma lista de dicionários contendo todas as magias previstas
    magias_completas = [{"Nome": nome, "Descricao": descricao} for nome, descricao in zip(nomes_pred, descricoes_pred)]
    
    return magias_completas

