# -------------------------------
# 🔹 Importación de librerías
# -------------------------------
# Herramientas para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
# Convierte texto a vectores TF-IDF (valores numéricos)
from sklearn.feature_extraction.text import TfidfVectorizer
# Modelo Naive Bayes Multinomial (bueno para clasificación de texto)
from sklearn.naive_bayes import MultinomialNB
# Permite encadenar pasos en un flujo (vectorización + modelo)
from sklearn.pipeline import Pipeline
# Métricas de evaluación del modelo
from sklearn.metrics import accuracy_score, f1_score
# Codifica etiquetas categóricas a números y viceversa
from sklearn.preprocessing import LabelEncoder
# Clase propia para manejar el dataset
from Data import Data
# Manejo de datos tabulares
import pandas as pd
# Para medir tiempos de ejecución
import time


# -------------------------------
# 🔹 Clase principal del modelo
# -------------------------------
class ModelBN:
    
    def __init__(self, params, csv_path="commands_dataset.csv"):
        """
        Constructor de la clase ModelBN.
        - params: parámetros que se le pasan al modelo Naive Bayes.
        - csv_path: ruta del dataset CSV que se va a cargar.
        """
        self.csv_path = csv_path
        self.name = "Naive_Bayes_Model"  # Nombre del modelo
        # Pipeline = flujo que transforma texto con TF-IDF y luego entrena el modelo
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),       # Convierte texto a números
            ("model", MultinomialNB(**params))  # Entrena modelo Naive Bayes
        ])
        # Carga de datos mediante la clase Data
        self.data = Data(csv_path)
        # Codificador para transformar etiquetas (strings) a números
        self.encoder = self.data.encoder
        


    # -------------------------------
    # 🔹 Entrenamiento del modelo
    # -------------------------------
    def train(self):
        """
        Entrena el modelo Naive Bayes usando el dataset.
        Retorna métricas y datos útiles del entrenamiento.
        """
        start_time_train = time.time()  # Guardamos el tiempo inicial
        
        # Cargar los textos y las etiquetas desde el CSV
        texts, labels = self.data.load_data()
        
        # Convertir las etiquetas de texto a valores numéricos
        labels_encoded = self.encoder.fit_transform(labels)
        
        # Dividir los datos en entrenamiento (70%) y prueba (30%)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=0.3, random_state=42
        )
        
        # Entrenamiento del modelo
        print("\n🧠 Entrenando el modelo Naive Bayes... Por favor espera ⏳")
        self.pipeline.fit(X_train, y_train)
        end_time_train = time.time()  # Tiempo final de entrenamiento
        
        # Predicciones sobre los datos de prueba
        y_pred = self.pipeline.predict(X_test)
        # Calcular métricas de desempeño
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        # Calcular tiempo total de entrenamiento
        train_time = (end_time_train - start_time_train)
        
        # Mostrar métricas de desempeño en consola
        print("\n✅ ---------- METRICAS DEL ENTRENAMIENTO ----------")
        print(f"📈 accuracy_score: {acc:.4f}")     # Qué tan bien predice el modelo
        print(f"🎯 F1_score: {f1:.4f}")            # Promedio ponderado entre precisión y recall
        print(f"⏱️ Tiempo de entrenamiento: {train_time:.4f}s\n")

        print("💾 Modelo entrenado y listo para usarse 🟢\n")

        # Retorna datos útiles del entrenamiento
        return y_pred, acc, f1, self.csv_path, train_time, X_train


    # -------------------------------
    # 🔹 Prueba del modelo con texto nuevo
    # -------------------------------
    def test(self, text):
        """
        Permite probar el modelo con una nueva entrada de texto.
        Retorna la predicción, la probabilidad y el tiempo de inferencia.
        """
        start_time = time.time()  # Inicio del cronómetro de predicción
        
        # Realizar predicción
        pred = self.pipeline.predict([text])
        # Obtener probabilidades de cada clase
        proba = self.pipeline.predict_proba([text])[0]

        # Decodificar la etiqueta numérica a su valor original (texto)
        pred_decoded = self.encoder.inverse_transform(pred)[0]
        end_time = time.time()
        # Calcular tiempo de inferencia (predicción)
        delay = end_time - start_time

        # Mostrar resultados en consola
        print("\n🧪 ---------- PRUEBA NUEVA ----------")
        print(f"📝 Texto ingresado: {text}")
        print(f"🤖 Predicción: {pred_decoded}")
        print(f"🔥 Confianza del modelo: {max(proba):.4f}")
        print(f"⚡ Tiempo de predicción: {delay:.4f} s\n")
        print("✅ Modelo probado con éxito.\n")
        
        # Retornar valores para posible registro o evaluación posterior
        return text, pred_decoded, proba, delay
