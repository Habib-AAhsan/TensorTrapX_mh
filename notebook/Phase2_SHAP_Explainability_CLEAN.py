# Phase 2: SHAP Explainability - Converted from Notebook to Script

#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load model and data
model = keras.models.load_model("production_model.keras")
df = pd.read_csv("../data/breast_cancer_synthetic_3k_cleaned_from_L3.csv")
X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
# SHAP DeepExplainer (works with TF/Keras models)
# explainer = shap.DeepExplainer(model, X_train[:100])
# shap_values = explainer.shap_values(X_test[:50])  # Limit for speed

# First define your explainer and calculate SHAP values
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_test[:50])

# Now you can print shap_values safely
i = 0  # Pick an index for explanation
print("shap_values[0][i].shape:", shap_values[0][i].shape)





#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
print("shap_values shape:", np.array(shap_values).shape)


print("X_test[i] shape:", X_test[i].shape)


#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
# SHAP summary plot (shows global feature impact)
shap.summary_plot(shap_values, X_test[:50], feature_names=df.columns[:-1])



#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
shap.force_plot(
    explainer.expected_value,
    shap_values[i].squeeze(),     # or shap_values[i, :, 0]
    X_test[i],
    feature_names=df.columns[:-1]
)



#--------------------------------------------------------------------------------
# Code Cell
#--------------------------------------------------------------------------------
# Save SHAP individual prediction explanation as HTML and PNG

# Generate the force plot for a single instance
i = 1
shap_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[i].squeeze(),
    X_test[i],
    feature_names=df.drop("diagnosis", axis=1).columns
)

# Save as HTML
shap.save_html("shap_force_plot_instance1.html", shap_plot)

# Optional: Save as PNG using matplotlib (less precise, but works for static export)
# Note: This will only capture what shap.force_plot renders in Jupyter (not JavaScript plot)
# You may use Selenium for pixel-perfect PNG export if needed
