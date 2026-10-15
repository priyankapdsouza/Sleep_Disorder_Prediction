import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------------
# 1. LOAD DATA
# -----------------------------

# Healthy vs Unhealthy (binary)
data_hu = np.loadtxt('../dataa/healthy_unhealthy1.csv', delimiter=',')
X_hu = data_hu[:, :1024].reshape((data_hu.shape[0], 1024, 1))
y_hu = data_hu[:, -1]   # assumes labels are 0 = Healthy, 1 = Unhealthy

x_train_hu, x_test_hu, y_train_hu, y_test_hu = train_test_split(
    X_hu, y_hu, test_size=0.2, random_state=42, stratify=y_hu
)

# Disease multi-class
data_d = np.loadtxt('../dataa/disease1.csv', delimiter=',')  # <--- change filename if needed
X_d = data_d[:, :1024].reshape((data_d.shape[0], 1024, 1))
y_d = data_d[:, -1]   # assumes labels 0..4

# One-hot encode disease labels
# If sparse_output gives error, change to: OneHotEncoder(sparse=False)
encoder = OneHotEncoder(sparse_output=False)
y_d_encoded = encoder.fit_transform(y_d.reshape(-1, 1))

x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d_encoded, test_size=0.2, random_state=42, stratify=y_d
)

# Disease label names (make sure the order matches your real label encoding!)
diseases_labels = ['INS', 'NFLE', 'NARCO', 'RBD', 'PLM']


# -----------------------------
# 2. HEALTHY / UNHEALTHY MODELS
# -----------------------------

hu_models = {
    'cnn': 'model_cnn_HU_new.h5',
    'skip': 'model_skip_HU_new.h5',
    'lstm': 'model_lstm_HU_new.h5',
    'incep': 'model_incep_HU_new.h5'
}

hu_results = []

print("HU Model Evaluations:")
for name, file in hu_models.items():
    try:
        model = tf.keras.models.load_model(file)

        # Predict probabilities and convert to class 0/1
        y_pred_prob = model.predict(x_test_hu, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # Metrics
        acc = accuracy_score(y_test_hu, y_pred)
        prec = precision_score(y_test_hu, y_pred, average='binary')
        rec = recall_score(y_test_hu, y_pred, average='binary')
        f1 = f1_score(y_test_hu, y_pred, average='binary')
        cm = confusion_matrix(y_test_hu, y_pred)

        # Save results for table
        hu_results.append({
            'Model': name.upper(),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

        # Print metrics
        print(f"\n{name.upper()} HU Model:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        # Extra: classification report (optional)
        print("\nClassification Report:")
        print(classification_report(y_test_hu, y_pred,
              target_names=['Healthy', 'Unhealthy']))

        # Plot and save confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=['Healthy', 'Unhealthy'],
            yticklabels=['Healthy', 'Unhealthy']
        )
        plt.title(f'Confusion Matrix - {name.upper()} HU Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'cm_{name}_hu.png')
        plt.close()

    except Exception as e:
        print(f"{name.upper()}: Error loading or evaluating model - {e}")


# -----------------------------
# 3. DISEASE MODELS (MULTI-CLASS)
# -----------------------------

d_models = {
    'cnn': 'model_v4_new.h5',
    'skip': 'model_skip_D70_new.h5',
    'gru': 'model_gru_new.h5',
    'lstm': 'model_lstm_new.h5',
    'incep': 'model_incep_new.h5'
}

d_results = []

print("\nDisease Model Evaluations:")
for name, file in d_models.items():
    try:
        model = tf.keras.models.load_model(file)

        y_pred_prob = model.predict(x_test_d, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_d, axis=1)

        # Metrics (macro = average over classes equally)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)

        d_results.append({
            'Model': name.upper(),
            'Accuracy': acc,
            'Precision (macro)': prec,
            'Recall (macro)': rec,
            'F1 Score (macro)': f1
        })

        print(f"\n{name.upper()} Disease Model:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
              target_names=diseases_labels))

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=diseases_labels,
            yticklabels=diseases_labels
        )
        plt.title(f'Confusion Matrix - {name.upper()} Disease Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'cm_{name}_disease.png')
        plt.close()

    except Exception as e:
        print(f"{name.upper()}: Error loading or evaluating model - {e}")


# -----------------------------
# 4. SUMMARY TABLES FOR ALL MODELS
# -----------------------------

print("\n\n===== SUMMARY: HU MODELS =====")
hu_df = pd.DataFrame(hu_results)
print(hu_df)

print("\n\n===== SUMMARY: DISEASE MODELS =====")
d_df = pd.DataFrame(d_results)
print(d_df)


# -----------------------------
# 5. FIND BEST ACCURACY MODEL
# -----------------------------

# Best HU model based on accuracy
best_hu = hu_df.loc[hu_df['Accuracy'].idxmax()]
best_disease = d_df.loc[d_df['Accuracy'].idxmax()]

print("\n\n===== BEST MODEL: Healthy vs Unhealthy =====")
print(f"Model: {best_hu['Model']}")
print(f"Accuracy: {best_hu['Accuracy']:.4f}")
print(f"Precision: {best_hu['Precision']:.4f}")
print(f"Recall: {best_hu['Recall']:.4f}")
print(f"F1 Score: {best_hu['F1 Score']:.4f}")

print("\n===== BEST MODEL: Disease Classification =====")
print(f"Model: {best_disease['Model']}")
print(f"Accuracy: {best_disease['Accuracy']:.4f}")
print(f"Precision (Macro): {best_disease['Precision (macro)']:.4f}")
print(f"Recall (Macro): {best_disease['Recall (macro)']:.4f}")
print(f"F1 Score (Macro): {best_disease['F1 Score (macro)']:.4f}")
