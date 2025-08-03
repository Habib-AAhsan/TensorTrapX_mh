# ============================
# Breast Cancer Modeling Pipeline (Modular)
# ============================

# ----------- Imports -----------
#!/usr/bin/env python
# coding: utf-8

# ## ✅ Current Steps Covered in Notebook
# 
# - 📚 **Imports** – All required libraries are included (TensorFlow, Keras Tuner, etc.).
# - 🎯 **Feature Setup** – Correct extraction of features and target from the dataset.
# - 🧠 **Dense Model Building** – `model_builder` function wrapped for Keras Tuner.
# - 🔍 **Hyperparameter Tuning** – Keras Tuner `RandomSearch` used with dense layers.
# - 📊 **Results Evaluation** – Best hyperparameters printed, model trained, and evaluated.
# - ⚙️ **Production Ready** – Best model saved using the modern `.keras` format.
# - 📈 **Monitoring** – TensorBoard integration for training visualization and logging.
# - 📉 **Accuracy Plotting** – Training vs. validation accuracy and loss curves plotted.
# - 🧬 **CNN Extension** – CNN model builder function defined and tuned via Keras Tuner.
# 

# In[1]:


# 📚 Import necessary library
import pandas as pd

# 📂 Load the cleaned dataset (adjust the path if needed)
df = pd.read_csv('data/breast_cancer_synthetic_3k_cleaned_from_L3.csv')

# 🧠 Inspect columns to confirm the label column name
print(df.columns)


# In[2]:


# 🎯 Define features and target (modify if column name is 'target' instead)
X = df.drop(columns=['diagnosis'])  # drop the label column
y = df['diagnosis']                 # extract the label

# 📐 Confirm input shape for model input layer
input_shape = X.shape[1]

# 🧪 Optional: Check data dimensions
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")


# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


# 📌 Step 1: Define the core model builder with tunable hyperparameters
def model_builder(hp, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))

    # 🔧 Tunable units for the first dense layer
    hp_units = hp.Int('units', min_value=32, max_value=128, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

    # 🔧 Optional dropout
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))

    # 🔚 Output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # 🔧 Tunable learning rate
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model




# In[4]:


# 🧠 Step 2: Wrap model_builder so it conforms to Keras Tuner's expected signature
input_shape = X.shape[1]

def hypermodel_fn(hp):
    return model_builder(hp, input_shape=X.shape[1])

print(df.columns)


# In[5]:


# 🔍 Step 3: Initialize the RandomSearch tuner
tuner = kt.RandomSearch(
    hypermodel=hypermodel_fn,
    objective='val_accuracy',
    max_trials=10,                # 🧪 Try 10 different hyperparameter combinations
    executions_per_trial=2,       # 🌀 Train each combination twice
    directory='keras_tuner_dir',  # 💾 Folder to save tuning results
    project_name='breast_cancer_tuning'
)

# 🚀 Step 4: Run the hyperparameter search
tuner.search(X, y,
             epochs=50,
             validation_split=0.2,
             batch_size=32,
             verbose=1)



# In[6]:


# Print the top 10 trial results (ranked by objective metric)
tuner.results_summary(num_trials=10)


# In[7]:


# 🔍 Retrieve the Best Hyperparameters from the Tuner
# We get the top trial's hyperparameters (e.g. best units, dropout rate, learning rate)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 📋 Display the Best Hyperparameters
# This loop prints the chosen values from the best trial
print("Best hyperparameters:")
for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")

# 🏗️ Build the Best Model with the Selected Hyperparameters
# Using the best set of hyperparameters found by KerasTuner, we construct the model
best_model = tuner.hypermodel.build(best_hps)

# 🧠 Train the Best Model on Full Dataset (with Validation Split)
# We now train the best model using all available data, reserving 20% for validation
history = best_model.fit(X, y, validation_split=0.2, epochs=50, verbose=1)


# In[8]:


import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


# Save the best model in multiple formats

best_model.save("best_model.keras")
best_model.save("best_model.h5")

# 🧠 Extra Tip: To load it later
# from keras.models import load_model
# model = load_model("best_model.keras")



# In[10]:


# 🔀 Step: Create a test set using train_test_split
from sklearn.model_selection import train_test_split

# Use 20% of data for final testing
X_train_final, X_test, y_train_final, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train_final.shape}, Test set: {X_test.shape}")


# 🔁 Train best model on training set (not including test set)
history = best_model.fit(
    X_train_final, y_train_final,
    validation_split=0.2,
    epochs=50,
    verbose=1
)



# ✅ Evaluate on the final test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"🧪 Final Test Accuracy: {test_acc:.4f}")



# 
# ## ⚙️ Step 6: Prepare for Production
# 
# Now that we have a trained model, it's time to prepare it for deployment.
# 
# We'll do three things:
# 1. Export the model in TensorFlow's `SavedModel` format.
# 2. (Optional) Serve it with an API (e.g., FastAPI).
# 3. (Optional) Build a minimal frontend (e.g., Streamlit or Gradio).
# 
# ### ✅ Save Model in TensorFlow SavedModel Format
# 

# In[11]:


from pathlib import Path

# ✅ Add correct extension (.keras or .h5)
saved_model_path = Path("production_model.keras")  # or use "production_model.h5" if you prefer HDF5

best_model.save(saved_model_path)

print(f"✅ Model saved to: {saved_model_path.resolve()}")


# 
# ## 📈 Step 7: Add Monitoring with TensorBoard
# 
# Monitoring model training and inference performance is essential for long-term usage.
# TensorBoard can visualize accuracy, loss, and other metrics.
# 
# This cell shows how to use TensorBoard callback during training.
# 

# In[12]:


from datetime import datetime
from pathlib import Path
import tensorflow as tf

# TensorBoard logging directory
log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))

# Re-train with TensorBoard callback
history = best_model.fit(
    X, y,
    validation_split=0.2,
    epochs=50,
    callbacks=[tensorboard_cb],
    verbose=1
)


# TensorBoard logging directory
log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))

# Re-train with TensorBoard callback
model_with_logs = model_builder(best_hps, input_shape=X.shape[1])
model_with_logs.fit(X, y, validation_split=0.2, epochs=20, callbacks=[tensorboard_cb])


# ## Plot Training & Validation Accuracy/Loss
# 
# You can visualize training progress like this:

# In[13]:


import matplotlib.pyplot as plt

# Extract metrics from history
history_dict = history.history

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Train Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# 
# ## 🧬 Step 8: Future Work — Try Advanced Architectures
# 
# You can now explore more advanced deep learning models suited for tabular data, such as:
# 
# - **1D Convolutional Neural Networks (CNNs)**: For sequential feature extraction
# - **TabNet**: Deep learning architecture optimized for tabular data
# - **AutoML tools**: Like AutoKeras, H2O.ai, or Google's AutoML Tables
# - **Transfer learning**: Combine tabular + imaging data if needed
# 
# These can boost performance and offer richer model behavior.
# 
# > Tip: Wrap advanced architectures in `model_builder()` and re-run Keras Tuner.
# 

# 
# ## 🧠 CNN Model Training Results
# 
# After tuning hyperparameters, the best CNN model achieved outstanding validation performance.
# 
# **Best Hyperparameters**:
# - `filters`: 16
# - `units`: 96
# - `dropout`: 0.1
# - `learning_rate`: 0.01
# 
# **Sample of Training Performance**:
# ```plaintext
# Epoch 1/50 - val_accuracy: 0.8417 - val_loss: 0.3695
# ...
# Epoch 50/50 - val_accuracy: 0.8980 - val_loss: 0.2747
# ```
# 
# 📈 Validation Accuracy peaked at: **~92.5%**, which outperforms the earlier MLP model.
# 

# In[14]:


def cnn_model_builder(hp, input_shape):
    model = tf.keras.models.Sequential()

    # Reshape input for Conv1D
    model.add(tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)))

    # Hyperparameter-tuned Conv1D layer
    hp_filters = hp.Int('filters', min_value=16, max_value=128, step=16)
    model.add(tf.keras.layers.Conv1D(filters=hp_filters, kernel_size=3, activation='relu'))

    # Optional max pooling
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Flatten before dense
    model.add(tf.keras.layers.Flatten())

    # Tunable dense layer
    hp_units = hp.Int('units', min_value=32, max_value=128, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

    # Dropout regularization
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Learning rate tuning
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# In[15]:


# 🔍 Initialize Keras Tuner for CNN model using Random Search
cnn_tuner = kt.RandomSearch(
    # Use a lambda to pass both the hyperparameter object and input shape to the model builder
    hypermodel=lambda hp: cnn_model_builder(hp, input_shape=X.shape[1]),

    # The objective to optimize for is validation accuracy (higher is better)
    objective='val_accuracy',

    # Total number of different hyperparameter combinations to try
    max_trials=10,

    # Each trial is run multiple times to average out randomness
    executions_per_trial=2,

    # Directory to store tuning logs and results
    directory='keras_tuner_dir',

    # Project name to organize tuning results (subfolder)
    project_name='cnn_breast_cancer_tuning'
)

# 🚀 Start hyperparameter search
cnn_tuner.search(
    X,                # Feature matrix
    y,                # Target labels
    epochs=50,        # Train each trial for up to 50 epochs
    validation_split=0.2,  # Use 20% of data as validation set
    batch_size=32,    # Size of batches during training
    verbose=1         # Show progress output during training
)


# In[16]:


cnn_best_hps = cnn_tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest CNN Hyperparameters:")
for param in cnn_best_hps.values:
    print(f"{param}: {cnn_best_hps.get(param)}")

cnn_best_model = cnn_tuner.hypermodel.build(cnn_best_hps)
cnn_history = cnn_best_model.fit(X, y, validation_split=0.2, epochs=50, verbose=1)

# Save the CNN model
cnn_best_model.save("cnn_best_model.keras")



# ----------- Functions & Builders -----------


# ----------- Main Execution Flow -----------
