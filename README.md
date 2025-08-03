# 🧠 TensorTrapX^(mh): Deep Diagnostic Engine for Breast Cancer Resilience

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-ff69b4)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)
![Platform](https://img.shields.io/badge/Tested_on-macOS/Linux-informational)

> A full-stack, production-oriented breast cancer prediction system using traditional ML, deep neural networks, CNNs, hyperparameter tuning (Keras Tuner), and interpretability tools — all orchestrated with scalable design principles.

---

## 🚀 Project Summary

**TensorTrapX^(mh)** aims to combine powerful data pipelines, deep learning, and explainable AI for robust breast cancer diagnosis from structured (tabular) data. Inspired by real-world use cases in medical diagnostics, the system goes from raw data to deployment-ready APIs and dashboards.

---

## ✅ Features Implemented

- 📚 Data preprocessing & feature engineering
- 🔢 Baseline ML and Deep Learning models
- 🎯 Hyperparameter tuning using Keras Tuner
- 🧠 CNN model for improved feature extraction
- 📊 Metrics visualization (accuracy, loss)
- 💾 Saved model in `.keras` format
- 📈 TensorBoard monitoring
- 🔍 Training-validation plots
- 💬 Markdown-rich Jupyter Notebook with explanations

---

## 📁 Project Structure (Finalized)

## 📁 Project Structure

```
TensorTrapX_mh/
├── main.py                       # FastAPI app with prediction, explain, feedback endpoints
├── feedback_worker.py           # RQ worker to process asynchronous feedback jobs
├── setup.py                     # Setup for packaging (optional for distribution)
├── requirements.txt             # Final pip dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Clean git tracking rules
│
├── data/                        # Project artifacts and data visualizations
│   └── train-validation-loss.png
│
├── images/                      # Static images used in README, diagrams, SHAP/LIME
│   ├── ROC.png
│   ├── SHAP VALUE.png
│   ├── SHAPE VS LIME.png
│   ├── TensorTrapX Async Feedback Loop Architecture.png
│   └── ...
│
├── logs/                        # Feedback logs, if used
│   └── feedback_log.csv
│
├── model/                       # Trained and saved models
│   ├── best_model.h5
│   └── best_model.keras
│
├── notebook/                    # Jupyter notebooks for preprocessing, modeling, explainability
│   ├── 1_preprocessing_BreastCancer.ipynb
│   ├── 2_modeling_tf_BreatCancer.ipynb
│   ├── Phase2_Model_LIME_Explainability.ipynb
│   ├── Phase2_SHAP_Explainability_CLEAN.ipynb
│   ├── shap_force_plot_instance1.html
│   └── lime_explanation_instance_5.html
│
├── test_api_predict.py          # Python script to test /predict endpoint with 5 samples
├── test_explain.py              # Python script to test /explain endpoint
│
└── tensorboard/                 # TensorBoard logs (auto-generated if enabled)
```

---

## 🧪 Key Tech Stack

- **Languages**: Python 3.11+
- **Libraries**:
  - TensorFlow, Keras
  - Keras Tuner (hyperparameter tuning)
  - SHAP, LIME (model explainability)
  - Streamlit/FastAPI (dashboard)
  - Pandas, NumPy, Seaborn, Matplotlib

---

## 📊 Model Performance

| Model Type | Accuracy (Val/Test) | Tuning Used |
|------------|---------------------|-------------|
| Dense NN   | ~90.4%              | ✅ Keras Tuner |
| CNN Model  | ~92.5%              | ✅ Keras Tuner |

---

## 🔧 Getting Started

```bash
# Step 1: Create virtual env
python3.11 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run notebook or scripts
jupyter notebook
# OR
python scripts/main_pipeline.py
```

---

## 📈 Monitoring

- TensorBoard (`logs/`)
- Future: Prometheus + Grafana Integration (Phase 2+)

---

# 🌍 Future Roadmap

### ✅ Phase 1: MVP + CNN  
Core model built using **TensorFlow** with **Conv1D** layers and tuned with **Keras Tuner**.

### ✅ Phase 2: Model Explainability (SHAP, LIME)  
**SHAP** and **LIME** integrated for local and global explainability with comparison plots.

### ✅ Phase 3: Real-Time Prediction API (FastAPI)  
Asynchronous **FastAPI** app with **Redis + RQ**, supporting `/predict`, `/feedback`, and `/explain` endpoints, with **SQLite** logging.

### 🔜 Phase 4: Streamlit / Gradio Dashboard  
User-facing dashboard for real-time predictions, visual insights, and feedback loop.

### 🔜 Phase 5: Monitoring + Logging (Prometheus, Grafana)  
Metrics collection, visual dashboards, and performance monitoring for **MLOps observability**.

### 🔜 Phase 6: Dockerization + GitHub Actions CI/CD  
Containerization, deployment automation, and **GitHub Actions** for production readiness.


---


👨‍💻 Author

A Ahsan (HABIB)
Data Engineer & Researcher (Machine Learning & Security)




## 📜 License

MIT License © A Ahsan (HABIB)
