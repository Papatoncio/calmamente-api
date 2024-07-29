import os
import pandas as pd
import pickle
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from flask import Flask, jsonify, request, render_template, make_response
from flask_cors import CORS
from supabase import create_client, Client
from supabase.client import ClientOptions
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from PIL import Image as imgb64

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

# * Index
@app.route('/')
def index():
    return render_template('index.html')

# ! Reporte PDF
@app.route('/reporte')
def generatepdf():
    try:
        body = {"":""}

        # Si el body esta vació retornar todas las columnas
        if body is None or body == {}:
            return jsonify({"error": "No se encontraron valores para realizar la operación"}), 400

        idmodelo = 50

        # Verificar si se proporciono idmodelo
        if idmodelo is None or idmodelo == "":
            return jsonify({"error": "No se definió un modelo para el reporte."}), 400

        # Datos para el reporte
        datos = supabase.table("datos").select("*").order(column="tipo", desc=False).execute().data

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        styles = getSampleStyleSheet()
        story = []

        for dato in datos:
            # Titulo de cada elemento
            nombre = dato.get("nombre")
            story.append(Paragraph(f"<b>{nombre}</b>", styles["Title"]))
            story.append(Spacer(1, 6))

            tipo = dato.get("tipo")
            valor = dato.get("valor")

            if tipo == "1":
                # Convertir la cadena a una lista
                lista = ast.literal_eval(valor)
                table_data = []
                for i in range(0, len(lista), 2): # Manejo de arreglos de texto
                    row = lista[i:i+2]
                    if len(row) < 2:
                        row.append('')  # Añadir una celda vacía si la fila tiene menos de 2 elementos
                    table_data.append(row)

                # Crear la tabla
                table_data_paragraphs = [[Paragraph("- " + cell, styles['BodyText']) for cell in row] for row in table_data]

                table = Table(table_data_paragraphs, colWidths='*')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.azure),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.azure),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))

                story.append(table)
                story.append(Spacer(1, 12))
            elif tipo == "2":
                img_data = base64.b64decode(valor)
                img_buffer = BytesIO(img_data)
                img = Image(img_buffer, 400, 300)
                story.append(img)

        doc.build(story)
        buffer.seek(0)

        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'inline; filename=output.pdf'

        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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
@app.route('/respuestas/csv', methods=["POST"])
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

        X = df[modeloglobal.feature_names_in_]

        # Verificar si modeloglobal está inicializado
        if modeloglobal is None:
            return jsonify({"error": "El modelo no ha sido cargado."}), 400

        prediccion = modeloglobal.predict(X)[0]

        # Retorna la predicción con los datos insertados
        return jsonify({ "mensaje" : "Predicción realizada correctamante",
                        "prediccion" : str(prediccion) }), 200
    except Exception as e:
        # Retorna el error
        return jsonify({ "error" : str(e) }), 400

# ! Peticiones Modelos
# * Obtener Modelos
@app.route('/modelos')
def getallmodelos():
    try:
        query = supabase.table("modelos").select("id, created_at, tipo, parametros, precision, principal")

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
        
        nombre = body.get("nombre")
        if nombre is None or nombre == "":
            return jsonify({"error": "No se especifico el nombre para el modelo"}), 400
                
        dataset = body.get("dataset")
        if dataset is None or dataset == "":
            return jsonify({"error": "No se proporciono un set de datos"}), 400
        
        # Transformar texto a formato csv
        file_path = 'C:/Users/juanp/Downloads/Datos_Niveles_Ansiedad.csv'
        df = pd.read_csv(file_path)

        tipo = body.get("tipo")
        if tipo is None or tipo == "":
            return jsonify({"error": "No se especifico el tipo de algoritmo"}), 400
        
        if tipo == "kmeans":
            modelo = KMeans(**parametros)
        elif tipo == "gaus":
            modelo = GaussianMixture(**parametros)
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
            "nombre" : nombre,
            "parametros": parametros,
            "precision": precision,
            "modelo": modelo_base64  # Guardar la cadena Base64
        }

        idmodelo = supabase.table("modelos").insert(data).execute().data[0].get("id")

        if tipo == "kmeans":
            clusterskmeans(idmodelo, modelo, X)
            metodocodokmeans(idmodelo, X)
        elif tipo == "gaus":
            # sdasd
            print("gaus")
        else:
            return jsonify({"error": "Tipo de algoritmo no soportado"}), 400

        columnas = list(X.columns)

        if columnas is None or columnas == []:
            return jsonify({"error": "No se encontraron columnas"}), 400

        # # Guardar columnas del dataset
        columnasdataset(idmodelo, columnas)

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

        cargarmodelo()

        return jsonify({"mensaje": "Modelo actualizado correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# * Actualizar Modelo
@app.route('/modelos/cargado')
def checkmodelo():
    try:
        print(modeloglobal.__getstate__())
        return jsonify({"mensaje": "Modelo actualizado correctamente", "modelo" : {
            "clase" : str(modeloglobal.__class__)
        } }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# * Guardar datos para reporte pdf
def columnasdataset(idmodelo, columnas):
    try:
        guardardatosreporte("Columnas Dataset", columnas, "1", idmodelo)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# * Gráficas Kmeans
def metodocodokmeans(idmodelo, data):
    try:
        # Gráfico de Elbow
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
            
        plt.figure(figsize=(8, 6))
        plt.plot(K, distortions, 'bo-')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('Inercia')
        plt.title('Gráfico de Elbow')

        guardardatosreporte("Método del Codo Kmeans", plt, "2", idmodelo)
        plt.close("all")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def clusterskmeans(idmodelo, modelo, data):
    try:
        print("---")
        # Centroides de los clusters
        centroids = modelo.cluster_centers_
        print("---")
        # Etiquetas de los clusters
        labels = modelo.labels_
        print("---")
        # Gráfico de dispersión con clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis', alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroides')
        plt.legend()
        plt.title('Gráfico de Dispersión con Clusters')
        plt.colorbar(scatter, label='Cluster')
        plt.show()
        print("---")
        guardardatosreporte("Clusters Kmeans", plt, "2", idmodelo)
        plt.close("all")
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
# ! Gráficas gauss

def guardardatosreporte(nombre, dato, tipo, idmodelo):
    try:        
        if nombre is None or nombre == "":
            return jsonify({"error": "No se establecio un nombre"}), 400
        
        if dato is None or dato == "":
            return jsonify({"error": "No se establecio un dato"}), 400
        
        if tipo is None or tipo == "":
            return jsonify({"error": "No se establecio un tipo"}), 400
        
        if idmodelo is None or idmodelo == "":
            return jsonify({"error": "No se establecio un idmodelo"}), 400

        if tipo == "1": # Gráficas
            valor = str(dato)
        elif tipo == "2": # Texto
            buffer = BytesIO()
            dato.savefig(buffer, format='png')
            buffer.seek(0)
            valor = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()            

        data = {
            "nombre" : nombre,
            "valor" : valor,
            "id_modelo" : idmodelo,
            "tipo" : tipo
        }

        supabase.table("datos").insert(data).execute()

        return jsonify({"mensaje": "Datos guardados correctamente" }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# * Ejecución del servidor (Siempre debe ir al final de la configuración)
if __name__ == "__main__":
    with app.app_context():
        cargarmodelo()
    app.run(debug=True)