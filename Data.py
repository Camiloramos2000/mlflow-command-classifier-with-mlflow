import unicodedata
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Data:
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.encoder = LabelEncoder()

    def load_data(self):
        try:
            # --- 1️⃣ Verificar que haya ruta ---
            if not self.csv_path:
                raise ValueError("❌ No se proporcionó la ruta del archivo CSV.")

            # --- 2️⃣ Cargar dataset ---
            print(f"📂 Cargando dataset desde: {self.csv_path}")
            df = pd.read_csv(self.csv_path)

            # --- 3️⃣ Verificar columnas necesarias ---
            required_cols = {"text", "command", "intent"}
            if not required_cols.issubset(df.columns):
                raise KeyError(f"❌ Faltan columnas en el dataset. Se esperaban: {required_cols}")

            # --- 4️⃣ Crear columna combinada ---
            df["label"] = df["command"].astype(str) + " " + df["intent"].astype(str)
            df = df.drop(columns=["command", "intent"])

            texts = df["text"].values
            labels = df["label"].values

            # --- 5️⃣ Limpiar texto y codificar etiquetas ---
            texts_cleaned = [self.clean_text(text) for text in texts]
            labels_encoded = self.encoder.fit_transform(labels)

            print("✅ Dataset cargado y procesado correctamente.")
            return texts_cleaned, labels_encoded

        except FileNotFoundError:
            print(f"❌ No se encontró el archivo CSV en la ruta: {self.csv_path}")
            return None, None

        except pd.errors.EmptyDataError:
            print("⚠️ El archivo CSV está vacío o corrupto.")
            return None, None

        except KeyError as e:
            print(str(e))
            return None, None

        except Exception as e:
            print(f"⚠️ Error inesperado durante la carga de datos: {e}")
            return None, None

    def clean_text(self, text):
        try:
            if not isinstance(text, str):
                text = str(text)

            # Minúsculas
            text = text.lower()
            # Quitar tildes
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
            # Quitar símbolos, números y signos
            text = re.sub(r'[^a-zA-ZñÑ\s]', '', text)
            # Quitar espacios dobles
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        
        except Exception as e:
            print(f"⚠️ Error limpiando texto: {e}")
            return ""
