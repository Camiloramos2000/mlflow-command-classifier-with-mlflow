# 🧠 Command Classifier with Naive Bayes + MLflow

This project implements a **command and intent classifier** based on a **Multinomial Naive Bayes model**, with a full **MLOps workflow using MLflow** for tracking metrics, versioning, and controlled deployment of models.

The system processes text commands such as _"restart the system now"_ or _"shut down later"_, identifying their intent and type (e.g., immediate or scheduled action).

---

## 🚀 Technologies Used

- **Python 3.10+**
- **scikit-learn** — model training and TF-IDF vectorization
- **pandas** — data manipulation
- **MLflow** — experiment tracking, model registry, and version control
- **uuid** — unique run identifiers

---

## 📂 Project Structure

```
├── commands_dataset.csv         # Base dataset with texts, commands, and intents
├── Data.py                      # Data loading, cleaning, and encoding class
├── Model.py                     # Naive Bayes model with TF-IDF pipeline
├── main.py                      # Complete MLOps pipeline with MLflow integration
└── README.md                    # (This file)
```

---

## 🧩 Project Workflow

### 1️⃣ Data Loading and Cleaning (`Data.py`)

The `Data` class handles:
- Reading the CSV dataset (`commands_dataset.csv`)
- Cleaning text (lowercasing, removing accents/symbols, trimming spaces)
- Combining `command` and `intent` columns into one label
- Encoding labels with `LabelEncoder`

```python
data = Data("commands_dataset.csv")
texts, labels = data.load_data()
```

**Example output:**

```
📂 Loading dataset from: commands_dataset.csv
✅ Dataset successfully loaded and processed.
```

---

### 2️⃣ Model Training (`Model.py`)

The `ModelBN` class defines a **pipeline** using `TfidfVectorizer` and `MultinomialNB`.

During training:
- Splits data into train/test sets (70/30)
- Fits the model
- Calculates metrics (`accuracy`, `f1_score`, `train_time`)
- Returns metrics and useful data for MLflow

```python
modelbn = ModelBN(params={"alpha": 0.1})
y_pred, acc, f1, csv_path, train_time, X_train = modelbn.train()
```

**Example output:**

```
✅ ---------- TRAINING METRICS ----------
📈 accuracy_score: 0.9567
🎯 F1_score: 0.9576
⏱️ Training time: 0.9821s
```

---

### 3️⃣ MLflow Tracking (`main.py`)

The main script automates the **end-to-end MLOps pipeline**:

1. **Initialize an MLflow experiment**
   ```python
   mlflow.set_experiment("command_classifier_model")
   ```

2. **Log parameters and metrics**
   - Hyperparameters (`alpha`, `type_model`, `vectorizer`)
   - Model metrics (`accuracy`, `f1_score`, `train_time`)
   - Dataset artifact

   ```python
   mlflow.log_param("Model_name", modelbn.name)
   mlflow.log_metric("accuracy", acc)
   mlflow.log_artifact(csv_path)
   ```

3. **Test the model on a sample input**
   ```python
   text, pred_decoded, proba, delay = modelbn.test("restart the system now")
   mlflow.log_metric("prediction_probability-test", float(max(proba)))
   ```

4. **Register the model in the Model Registry**
   ```python
   model_info = mlflow.sklearn.log_model(
       sk_model=modelbn.pipeline,
       name=modelbn.name,
       signature=signature,
       input_example=input_example
   )
   ```

5. **Promote the model to “Production”**
   ```python
   mv = client.transition_model_version_stage(
       name=mv.name,
       version=mv.version,
       stage="production"
   )
   ```

---

## 🧱 Example Execution

```bash
$ py main.py
```

**Expected output:**
```
🌍 STARTING MLFLOW PROCESS
🔗 Current Tracking URI: file:///.../mlruns
🧠 TRAINING NAIVE BAYES MODEL
📊 Logging metrics and parameters to MLflow...
📦 REGISTERING MODEL IN MLflow MODEL REGISTRY
🚀 TRANSITIONING MODEL TO STAGE: PRODUCTION
✅ Model successfully moved to 'Production'.
🎉 END OF FULL PROCESS
```

---

## 📊 MLflow UI Visualization

When the project runs, MLflow creates an `mlruns/` directory.  
You can visualize the results with:

```bash
mlflow ui
```

Then open in your browser:
> http://localhost:5000

Here you’ll find:
- Metric plots for each run  
- Model versions and stages  
- Logged parameters  
- Uploaded artifacts (e.g., datasets)

---

## 🧪 Quick Inference

After training, you can test the model directly:

```python
from Model import ModelBN

model = ModelBN(params={"alpha": 0.1})
text, pred, proba, delay = model.test("shut down system now")

print(pred)  # → "shutdown immediate"
```

---

## 🧠 Key Concepts

| Component | Description |
|------------|--------------|
| **TF-IDF Vectorizer** | Converts text into numerical vectors based on word frequency. |
| **MultinomialNB** | Probabilistic model suitable for text classification. |
| **MLflow Tracking** | Tracks runs, parameters, metrics, and artifacts. |
| **Model Registry** | Manages model versions and stages (`Staging`, `Production`). |

---

## 🧩 Future Improvements

- Expand dataset with more commands and synonyms.  
- Improve classification accuracy.

---

## 👨‍💻 Author

**Camilo Ramos Cotes**  
Full Stack & MLOps Developer  
📧 camutoxlive20@gmail.com 
🧩 _“From commands to context: applied intelligence.”_

---

## 🏁 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with proper author attribution.
