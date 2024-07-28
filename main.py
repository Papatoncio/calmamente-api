import os
import pandas as pd
import pickle
import base64
from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client, Client
from supabase.client import ClientOptions
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn import metrics, svm
from io import StringIO

# * Instanciación de la app
app = Flask(__name__)

# * Agrega soporte CORS para todos los endpoints, todos los origenes
CORS(app)

# ! Variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# * Conexión a base de datos en supabase
url: str = SUPABASE_URL
key: str = SUPABASE_KEY
supabase: Client = create_client(url, key,
  options=ClientOptions(
    postgrest_client_timeout=10,
    storage_client_timeout=10,
    schema="public",
  ))

# ! Peticiones Respuestas
# * Obtener todas las respuestas
@app.route('/respuestas')
def getallrespuestas():
    try:
        query = supabase.table("respuestas").select("*")

        # Retorna los resultados de la consulta
        return jsonify({ "mensaje" : "Datos obtenidos correctamente", "rows" : str(query.execute().data) }), 200
    except Exception as e:
        # Retorna el error
        return jsonify({ "error" : str(e) }), 400

# * Obtener datos para exportar a excel
@app.route('/respuestas/csv')
def exportcsv():
    try:
        query = supabase.table("respuestas")
        body = request.get_json()

        # Si el body esta vació retornar todas las columnas
        if body is None or body == {}:
            return jsonify({ "error" : "No se encontraron valores a para realizar la operación" }), 400

        columnas = body.get("columnas")

        # Si hay columnas por las que filtrar, se agregan a la query
        if columnas:
            query = query.select(columnas)
        else:
            query = query.select("*")

        filtros = body.get("filtros")

        # Si hay filtros, se agregan a la query
        if filtros:
                for filtro in filtros:
                    if filtro.get("tipo") == "eq":
                        query = query.eq(filtro.get("columna"), filtro.get("valor"))                       
                    elif filtro.get("tipo") == "gte":
                        query = query.gte(filtro.get("columna"), filtro.get("valor"))
                    elif filtro.get("tipo") == "lt":
                        query = query.lt(filtro.get("columna"), filtro.get("valor"))

        # Regresar datos filtrados
        return jsonify({ "mensaje" : "Datos consultados correctamente", "rows" : str(query.csv().execute()) }), 200
    except Exception as e:
        # Retorna el error
        return jsonify({ "error" : str(e) }), 400

# * Insertar respuesta
@app.route('/respuestas', methods=["POST"])
def insertrespuesta():
    try:
        body = request.get_json()
        
        # Si el body esta vació retornar mensaje
        if body is None or body == {}:
            return jsonify({ "error" : "No se encontraron valores a insertar" }), 400

        supabase.table("respuestas").insert(body).execute()

        # Convertir el Json a un dataframe
        df = pd.DataFrame([body])

        y = df['Nivel de Ansiedad']
        X = df.drop(columns=['Nivel de Ansiedad'])

        # Imprimir las columnas de X que se usaron para ajustar el modelo
        print("Columnas usadas para entrenar el modelo:", modeloglobal.feature_names_in_)

        # Imprimir las columnas del nuevo DataFrame
        print("Columnas en el DataFrame de entrada:", X.columns.tolist())

        # Verificar si hay coincidencias
        if set(X.columns) != set(modeloglobal.feature_names_in_):
            print("Columnas que no coinciden:", set(X.columns).symmetric_difference(set(modeloglobal.feature_names_in_)))

        # Verificar si modeloglobal está inicializado
        if modeloglobal is None:
            return jsonify({"error": "El modelo no ha sido cargado."}), 400

        predicciones = modeloglobal.predict(X)
        score = metrics.adjusted_rand_score(y, predicciones)

        # Retorna la predicción con los datos insertados
        return jsonify({ "mensaje" : "Predicción realizada correctamante",
                        "score" : score }), 200
    except Exception as e:
        # Retorna el error
        return jsonify({ "error" : str(e) }), 400

# ! Peticiones Modelos
# * Obtener Modelos
@app.route('/modelos')
def getallmodelos():
    try:
        query = supabase.table("modelos").select("*")

        # Retorna los resultados de la consulta
        return jsonify({ "mensaje" : "Datos consultados correctamente", "rows" : str(query.execute().data) }), 200
    except Exception as e:
        # Retorna el error
        return jsonify({ "error" : str(e) }), 400

# ! Crear Modelo - Falta revisar formato con el que llega desde el front el dataset
@app.route('/modelos', methods=["POST"])
def insertmodelo():
    try:
        body = request.get_json()

        if body is None or body == {}:
            return jsonify({"error": "No se encontraron valores a insertar"}), 400

        parametros = body.get("parametros")
        if parametros is None or parametros == "":
            return jsonify({"error": "No se proporcionaron parámetros"}), 400
                
        dataset = body.get("dataset")
        if dataset is None or dataset == "":
            return jsonify({"error": "No se proporciono un set de datos"}), 400
        
        # Transformar texto a formato csv
        csv = StringIO(dataset)
        df = pd.read_csv(csv)

        tipo = body.get("tipo")
        if tipo is None or tipo == "":
            return jsonify({"error": "No se especifico el tipo de algoritmo"}), 400
        
        if tipo == "kmeans":
            modelo = KMeans(**parametros)
        elif tipo == "svc":
            modelo = svm.SVC(**parametros)
        else:
            return jsonify({"error": "Tipo de algoritmo no soportado"}), 400

        if modelo is None:
            return jsonify({"error": "Ocurrió un error al crear el modelo"}), 400
        
        y = df['Nivel de Ansiedad']
        X = df.drop(columns=['Nivel de Ansiedad'])

        modelo.fit(X)
        predicciones = modelo.predict(X)
        precision = metrics.adjusted_rand_score(y, predicciones)

        # Serializar el modelo y convertirlo a Base64
        modelo_bytes = pickle.dumps(modelo)
        modelo_base64 = base64.b64encode(modelo_bytes).decode('utf-8')  # Convertir a cadena

        data = {
            "tipo": tipo,
            "parametros": parametros,
            "precision": precision,
            "modelo": modelo_base64  # Guardar la cadena Base64
        }

        supabase.table("modelos").insert(data).execute()

        return jsonify({ "mensaje": "Modelo guardado correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# * Cargar Modelo
@app.route('/modelos/cargar')
def cargarmodelo():
    # * Instanciación de modelo global
    global modeloglobal

    try:
        query = supabase.table("modelos").select("modelo").eq("principal", True)
        modelo_base64 = query.execute().data[0].get("modelo")  # Obtener la cadena Base64

        # Convertir de Base64 a bytes
        modelobytes = base64.b64decode(modelo_base64)

        modeloglobal = pickle.loads(modelobytes)  # Cargar el modelo

        if modeloglobal is None:
            return jsonify({"error": "No se ha cargado un modelo"}), 400

        return jsonify({"mensaje": "Modelo cargado correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# * Eliminar Modelo
@app.route('/modelos', methods=["DELETE"])
def deletemodelo():
    try:
        body = request.get_json()

        if body is None or body == {}:
            return jsonify({"error": "No se encontraron valores para realizar la operación"}), 400
        
        idmodelo = body.get("id")

        if idmodelo is None or idmodelo == "":
            return jsonify({"error": "No se encontro el modelo"}), 400

        supabase.table("modelos").delete().eq("id", idmodelo).execute()

        return jsonify({"mensaje": "Modelo eliminado correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# * Actualizar Modelo
@app.route('/modelos', methods=["PUT"])
def updatemodelo():
    try:
        body = request.get_json()

        if body is None or body == {}:
            return jsonify({"error": "No se encontraron valores para realizar la operación"}), 400
        
        idmodelo = body.get("id")

        if idmodelo is None or idmodelo == "":
            return jsonify({"error": "No se encontro el modelo"}), 400

        # Establecer atributo principal de resto de modelos como false
        supabase.table("modelos").update({ "principal" : False }).neq("id", idmodelo).execute()

        # Establecer atributo principal de modelo como true
        supabase.table("modelos").update({ "principal" : True }).eq("id", idmodelo).execute()

        return jsonify({"mensaje": "Modelo actualizado correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400    

# * Ejecución del servidor (Siempre debe ir al final de la configuración)
if __name__ == "__main__":
    with app.app_context():
        cargarmodelo()
    app.run(debug=True)