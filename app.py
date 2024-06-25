import random
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Datos de ejemplo para simular la predicción aproximada
datos_ejemplo = [
    {"carat": 0.30, "cut": "Ideal", "color": "E", "clarity": "SI1", "depth": 62.1, "table": 58.0, "x": 4.27, "y": 4.29, "z": 2.66, "price": 499},
    {"carat": 0.33, "cut": "Premium", "color": "G", "clarity": "IF", "depth": 60.8, "table": 58.0, "x": 4.42, "y": 4.46, "z": 2.70, "price": 984},
    {"carat": 0.90, "cut": "Very Good", "color": "E", "clarity": "VVS2", "depth": 62.2, "table": 60.0, "x": 6.04, "y": 6.12, "z": 3.78, "price": 6289},
    {"carat": 0.42, "cut": "Ideal", "color": "F", "clarity": "VS1", "depth": 61.6, "table": 56.0, "x": 4.82, "y": 4.80, "z": 2.96, "price": 1082},
    {"carat": 0.31, "cut": "Ideal", "color": "F", "clarity": "VVS1", "depth": 60.4, "table": 59.0, "x": 4.35, "y": 4.43, "z": 2.65, "price": 779}
]

# Cargar datos
data_df = pd.read_csv("https://raw.githubusercontent.com/meander02/dtaset/main/cubic_zirconia.csv")

# Definir las categorías posibles para cada característica categórica
cuts = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
colors = ['J', 'I', 'D', 'H', 'F', 'E', 'G']
clarities = ['I1', 'IF', 'VVS1', 'VVS2', 'VS1', 'SI2', 'VS2', 'SI1']

# Configurar ColumnTransformer para aplicar OneHotEncoder solo a las columnas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories=[cuts, colors, clarities]), ['cut', 'color', 'clarity'])
    ], remainder='passthrough'
)

# Entrenar el preprocesador con los datos de ejemplo
df = pd.DataFrame(datos_ejemplo)
X_transformed = preprocessor.fit_transform(df[['cut', 'color', 'clarity']])
y = df['price']

# Entrenar el modelo (ejemplo con RandomForestRegressor)
model = RandomForestRegressor()
model.fit(X_transformed, y)

# Ruta principal para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que los datos se están enviando correctamente
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    # Convertir los valores de cadena a números flotantes y limpiar los datos
    try:
        carat = float(data['carat'].strip())  # Eliminar espacios adicionales
        depth = float(data['depth'].strip())
        table = float(data['table'].strip())
        x = float(data['x'].strip())
        y = float(data['y'].strip())
        z = float(data['z'].strip())
    except ValueError as e:
        return jsonify({'error': 'Los campos numéricos deben ser valores numéricos válidos.'}), 400

    # Preparar los datos de entrada para la predicción
    data_input = pd.DataFrame({
        'carat': [carat],
        'cut': [data['cut']],
        'color': [data['color']],
        'clarity': [data['clarity']],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    # Transformar datos de entrada usando el preprocesador entrenado
    data_transformed = preprocessor.transform(data_input)

    # Hacer predicción con el modelo
    prediction = model.predict(data_transformed)

    # Buscar en el dataset original si hay un registro similar y obtener su precio
    similar_data = data_df[
        (data_df['carat'] == carat) &
        (data_df['cut'] == data['cut']) &
        (data_df['color'] == data['color']) &
        (data_df['clarity'] == data['clarity']) &
        (data_df['depth'] == depth) &
        (data_df['table'] == table) &
        (data_df['x'] == x) &
        (data_df['y'] == y) &
        (data_df['z'] == z)
    ]

    if not similar_data.empty:
        similar_price = similar_data.iloc[0]['price']
        # Calcular un nearby_price basado en el precio similar
        nearby_price = similar_price + random.uniform(-1, 2)  # Ajusta el rango según necesites
        nearby_price = round(nearby_price, 2)  # Redondear a dos decimales

        result = {
            'prediction': float(prediction[0]),
            'nearby_price': nearby_price
        }
    else:
        result = {
            'prediction': float(prediction[0]),
            'nearby_price': None
        }

    # Devolver el precio predicho y el precio cercano encontrado (si existe)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
