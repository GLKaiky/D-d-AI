# Importar a função do módulo magias
from IA import obter_magias_por_nivel_e_classe
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import joblib
import base64

app = Flask(__name__)
CORS(app)

model_path = "/home/glkaiky/Desktop/plmg-cc-ti2-2024-1-g06-d-deas/Codigo/src/main/python/Sistema_Inteligente/Models"
feature_columns = joblib.load(f"{model_path}/feature_columns.pkl")

@app.route('/recomendar', methods = ['POST'])
def recomendar():
    data = request.get_json()
    classe = data['nome_classe']
    nivel = data.get('nivel') 
    recomendacoes = obter_magias_por_nivel_e_classe(nivel, classe, feature_columns, model_path)

    return jsonify({'recomendacoes': recomendacoes})

if __name__ == '__main__':
    app.run(debug=True, port=9090)

