# ===========================================================
# 📦 IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ===========================================================

import mlflow                              # Gestión del ciclo de vida del modelo
from mlflow import MlflowClient            # Cliente para interactuar con el servidor MLflow
import uuid                                # Genera identificadores únicos (IDs)
import pandas as pd                        # Manipulación de datos
from Model import ModelBN                  # Clase personalizada del modelo Naive Bayes
from mlflow.entities.model_registry import RegisteredModel


# ===========================================================
# 🧩 FUNCIÓN AUXILIAR PARA MOSTRAR INFO DE VERSIONES DE MODELOS
# ===========================================================

def print_model_version_info(mv):
    """Muestra información detallada de una versión del modelo en MLflow."""

    print(f"🧾 Name: {mv.name}")
    print(f"🔢 Version: {mv.version}")
    print(f"📝 Description: {mv.description}")
    print(f"📦 Stage: {mv.current_stage}\n")



# ===========================================================
# 🚀 MAIN (PUNTO DE ENTRADA DEL PROYECTO)
# ===========================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("🌍  INICIO DEL PROCESO CON MLFLOW")
    print("="*80 + "\n")



    # -------------------------------------------
    # 🌍 CONFIGURACIÓN DE MLFLOW
    # -------------------------------------------
    print(f"🔗 Tracking URI actual: {mlflow.get_tracking_uri()}\n")

    run_name = f"C.C.M-Run-{uuid.uuid4()}"
    mlflow.set_experiment("command_classifier_model")



    # -------------------------------------------
    # 🧠 ENTRENAMIENTO DEL MODELO
    # -------------------------------------------
    with mlflow.start_run(run_name=run_name) as run:

        client = MlflowClient()
        
        params = {"alpha": 0.1}

        mlflow.log_params(params)

        print("="*80)
        print("🧠 ENTRENANDO MODELO NAIVE BAYES")
        print("="*80 + "\n")

        modelbn = ModelBN(params=params)
        y_pred, acc, f1, csv_path, train_time, X_train = modelbn.train()

        print("\n📊 Registrando métricas y parámetros en MLflow...\n")

        mlflow.log_param("Model_name", modelbn.name)
        mlflow.log_param("type_model", "Naive Bayes")
        mlflow.log_param("vectorizer", "TfidfVectorizer")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("train_time", train_time)
        mlflow.log_artifact(csv_path)

        print("✅ Modelo entrenado y evaluado correctamente.")
        print("📁 Métricas y dataset registrados en MLflow.\n")
        print("-"*80 + "\n")



        # -------------------------------------------
        # 🧪 PRUEBA DEL MODELO
        # -------------------------------------------
        print("="*80)
        print("🧩 PRUEBA DEL MODELO Y REGISTRO EN MLFLOW")
        print("="*80 + "\n")

        myOrder = "reiniciar el sistema ahora"

        text, pred_decoded, proba, delay = modelbn.test(myOrder)

        mlflow.log_param("test_text", text)
        mlflow.log_param("predicted_label-test", pred_decoded)
        mlflow.log_metric("prediction_probability-test", float(max(proba)))
        mlflow.log_metric("inference_time-test", float(delay))

        print("✅ Prueba completada correctamente.")
        print("🗂️ Resultados del test registrados en MLflow.\n")
        print("-"*80 + "\n")



        # -------------------------------------------
        # 💾 REGISTRO DEL MODELO EN EL MODEL REGISTRY
        # -------------------------------------------
        print("="*80)
        print("📦 REGISTRO DEL MODELO EN MLflow MODEL REGISTRY")
        print("="*80 + "\n")

        input_example = pd.DataFrame({"text": ["reiniciar el sistema"]})
        example_pred = modelbn.pipeline.predict(input_example)

        signature = mlflow.models.infer_signature(input_example, example_pred)

        model_info = mlflow.sklearn.log_model(
            sk_model=modelbn.pipeline,
            name=modelbn.name,
            signature=signature,
            input_example=input_example,
        )
        try:
            client.create_registered_model(modelbn.name, description="Modelo Naive Bayes para clasificación de comandos e intenciones.",
                tags={"version": "1.0", "type_model": "Naive Bayes", "created_by": "Camilo Ramos"})
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                print(f"⚠️ El modelo '{modelbn.name}' ya existe en el Model Registry. Usando el modelo existente.\n")
            else:
                raise e
            

        mv = client.create_model_version(
            name=modelbn.name,
            source=model_info.model_uri,
            run_id=run.info.run_id,
            tags ={"version": "1.0"}
        )
        
        mv = client.update_model_version(
            name=mv.name,
            version=mv.version,
            description="Versión inicial del modelo Naive Bayes para clasificación de comandos e intenciones.",
        )

        print("💾 Modelo registrado exitosamente en MLflow Model Registry.\n")
        print("-"*80 + "\n")
        
    

        # -------------------------------------------
        # 🚦 TRANSICIÓN DEL MODELO A “PRODUCTION”
        # -------------------------------------------
        print("="*80)
        print("🚀 TRANSICIÓN DEL MODELO A STAGE: PRODUCTION")
        print("="*80 + "\n")
        
        print("📋 Información actual del modelo:")
        mv = client.get_model_version(name=mv.name, version=mv.version)
        print_model_version_info(mv)
        
        print("🚦 Transicionando el modelo a 'Production'...")
        mv = client.transition_model_version_stage(
            name=mv.name,
            version=mv.version,
            stage="production",
            archive_existing_versions=False,
        )
        
        print("🔄 Estableciendo 'Production' como alias del modelo...")
        client.set_registered_model_alias(
            name=mv.name,
            alias="Production",
            version=mv.version
        )

        print("💾✅Información del modelo cambiado exitosamente:")
        print("\n" + "="*70)
        print("📋  INFORMACIÓN DE VERSIÓN DEL MODELO (CAMBIADO A 'Production')")
        print("="*70)
        print_model_version_info(mv)
        print("✅ Modelo movido a 'Production' exitosamente.\n")
        print("-"*80 + "\n")
        # -------------------------------------------
        # 🏁 FIN DEL PROCESO
        # -------------------------------------------
        print("="*80)
        print("🎉 FIN DEL PROCESO COMPLETO")

