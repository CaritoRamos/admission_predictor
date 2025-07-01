from flask import Flask, render_template, request
import joblib
import pandas as pd

# 🔁 IMPORTANTE: Importa tu clase personalizada antes de cargar el modelo
from custom_transformers import softMax

app = Flask(__name__)

# Cargar modelo y transformador
modelo = joblib.load('model.pkl')
preprocessor = joblib.load('transformador.pkl')

# Leer carreras desde el archivo
with open("carreras.txt", encoding="utf-8") as f:
    lista_carreras = sorted([line.strip() for line in f if line.strip()])

@app.route('/')
def index():
    return render_template('index.html', carreras=lista_carreras)

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "CARRERA": request.form['carrera'],
        "PUNTAJE": float(request.form['puntaje']),
        "PERIODO": request.form['periodo'],
        "NRO_POSTULACION": int(request.form['nro_postulacion']),
        "SEXO": request.form['sexo']
    }

    df = pd.DataFrame([data])
    x_transformed = preprocessor.transform(df)
    pred = modelo.predict(x_transformed)
    resultado = int(pred[0])

    mensaje = (
        "✅ ¡Felicidades! Es probable que INGRESES a la universidad."
        if resultado == 1 else
        "❌ Con los valores ingresados es probable que NO ingreses. ¡Sigue intentando!"
    )

    return render_template('resultado.html', mensaje=mensaje)
