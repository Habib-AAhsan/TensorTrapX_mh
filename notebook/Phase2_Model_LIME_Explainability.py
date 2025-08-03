#!/usr/bin/env python
# coding: utf-8

# # 🧠 Phase 2: Model Explainability using SHAP and LIME
# 
# Understand why your model makes certain predictions — especially important for medical applications like breast cancer prediction.

# In[4]:


import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lime
import lime.lime_tabular as lime_tabular



# ## 📦 Load Pretrained Model and Dataset

# In[5]:


# Load model
model = tf.keras.models.load_model("best_model.keras")

# Load and prepare data (update this path as needed)
df = pd.read_csv("../data/breast_cancer_synthetic_3k_cleaned_from_L3.csv")
X = df.drop('diagnosis', axis=1).to_numpy()
y = df['diagnosis'].to_numpy()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 🔍 SHAP Explainability

# In[6]:


# KernelExplainer for Deep Learning Models
explainer = shap.Explainer(model, X_train[:100])  # sample to reduce computation
shap_values = explainer(X_test[:50])

# Summary plot
shap.summary_plot(shap_values, X_test[:50], feature_names=df.columns[:-1])


# ## 🌿 LIME Explainability

# ### 🔍 LIME Explanation for Individual Prediction
# 
# We wrap `model.predict()` to return both class probabilities because LIME expects predictions in `[p(class 0), p(class 1)]` format. This is common when using sigmoid activations in binary classification.
# 

# In[7]:


# LIME setup
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=df.columns[:-1],
    class_names=["Benign", "Malignant"],
    mode="classification"
)

# ⚙️ Define a wrapper function to provide probabilities for both classes (needed by LIME)
def predict_proba_wrapper(x):
    """Wrap model.predict to return [1-p, p] for binary classification."""
    preds = model.predict(x)
    return np.hstack([1 - preds, preds])

# 🔍 Use LIME to explain a prediction
i = 5  # Pick any instance
exp = explainer_lime.explain_instance(
    X_test[i],                  # instance to explain
    predict_proba_wrapper,      # wrapped predictor
    num_features=10             # number of features to show
)

# 💬 save explanation in as HTML
exp.save_to_file("lime_explanation_instance_5.html")
print("✅ LIME explanation saved to lime_explanation_instance_5.html")


