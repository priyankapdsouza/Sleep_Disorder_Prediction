import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# 1. LOAD DATA
# -----------------------------

# Healthy vs Unhealthy (binary)
data_hu = np.loadtxt(
    r'E:\FinalyearProject\Sleep-disorder-Classification\dataa\healthy_unhealthy1.csv',
    delimiter=','
)
X_hu = data_hu[:, :1024].reshape((data_hu.shape[0], 1024, 1))
y_hu = data_hu[:, -1]  # 0 = Healthy, 1 = Unhealthy

_, x_test_hu, _, y_test_hu = train_test_split(
    X_hu, y_hu,
    test_size=0.2,
    random_state=42,
    stratify=y_hu
)

# Disease multi-class from both_phases.csv
data_d = np.loadtxt(
    r'E:\FinalyearProject\Sleep-disorder-Classification\dataa\both_phases.csv',
    delimiter=','
)
X_d = data_d[:, :1024].reshape((data_d.shape[0], 1024, 1))
y_d = data_d[:, -1]  # assume labels: 0..4 for INS, NFLE, NARCO, RBD, PLM

try:
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(sparse=False)

y_d_encoded = encoder.fit_transform(y_d.reshape(-1, 1))

_, x_test_d, _, y_test_d = train_test_split(
    X_d, y_d_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_d
)

diseases_labels = ['INS', 'NFLE', 'NARCO', 'RBD', 'PLM']

# -----------------------------
# 2. MODEL PATHS
# -----------------------------

# These are relative to GUI\ (where this script is)
hu_models = {
    'CNN_HU':   'model_cnn_HU_new.h5',
    'SKIP_HU':  'model_skip_HU_new.h5',
    'LSTM_HU':  'model_lstm_HU_new.h5',
}

d_models = {
    'CNN_DISEASE':   'model_v4_new.h5',
    'SKIP_DISEASE':  'model_skip_D70_new.h5',
    'GRU_DISEASE':   'model_gru_new.h5',
    'INCEP_DISEASE': 'model_incep_new.h5',
}

# -----------------------------
# 3. EVALUATE MODELS
# -----------------------------

hu_results = []
d_results = []

# One PDF to store all confusion matrices
pdf = PdfPages('confusion_matrices_report.pdf')

print("\n==============================")
print(" HEALTHY vs UNHEALTHY MODELS ")
print("==============================")

for name, model_file in hu_models.items():
    try:
        print(f"\n>>> Evaluating {name} ({model_file})")
        model = tf.keras.models.load_model(model_file)

        y_pred_prob = model.predict(x_test_hu, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        acc = accuracy_score(y_test_hu, y_pred)
        prec = precision_score(y_test_hu, y_pred, zero_division=0)
        rec = recall_score(y_test_hu, y_pred, zero_division=0)
        f1 = f1_score(y_test_hu, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_hu, y_pred)

        hu_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_hu, y_pred,
                                    target_names=['Healthy', 'Unhealthy']))

        # Plot CM
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Unhealthy'],
            yticklabels=['Healthy', 'Unhealthy']
        )
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        img_path = f'cm_{name}.png'
        plt.savefig(img_path)
        pdf.savefig()   # add this page to PDF
        plt.close()
        print("Saved confusion matrix as:", img_path)

    except Exception as e:
        print(f"❌ Error evaluating {name}: {e}")

print("\n==============================")
print("       DISEASE MODELS        ")
print("==============================")

y_true_d = np.argmax(y_test_d, axis=1)

for name, model_file in d_models.items():
    try:
        print(f"\n>>> Evaluating {name} ({model_file})")
        model = tf.keras.models.load_model(model_file)

        y_pred_prob = model.predict(x_test_d, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(y_true_d, y_pred)
        prec = precision_score(y_true_d, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true_d, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true_d, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true_d, y_pred)

        d_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (macro)': prec,
            'Recall (macro)': rec,
            'F1 Score (macro)': f1
        })

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision (macro): {prec:.4f}")
        print(f"Recall (macro)   : {rec:.4f}")
        print(f"F1 Score (macro) : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_d, y_pred,
                                    target_names=diseases_labels))

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=diseases_labels,
            yticklabels=diseases_labels
        )
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        img_path = f'cm_{name}.png'
        plt.savefig(img_path)
        pdf.savefig()   # add page to PDF
        plt.close()
        print("Saved confusion matrix as:", img_path)

    except Exception as e:
        print(f"❌ Error evaluating {name}: {e}")

# Close PDF after all figures
pdf.close()
print("\n📄 Saved all confusion matrices into: confusion_matrices_report.pdf")

# -----------------------------
# 4. SAVE METRICS TO CSV
# -----------------------------
try:
    import pandas as pd

    hu_df = pd.DataFrame(hu_results)
    d_df = pd.DataFrame(d_results)

    hu_df.to_csv('hu_model_results.csv', index=False)
    d_df.to_csv('disease_model_results.csv', index=False)

    print("\n💾 Saved HU metrics to: hu_model_results.csv")
    print("💾 Saved Disease metrics to: disease_model_results.csv")

    print("\n===== SUMMARY: HEALTHY vs UNHEALTHY MODELS =====")
    print(hu_df)

    if not hu_df.empty:
        best_hu = hu_df.loc[hu_df['Accuracy'].idxmax()]
        print("\n⭐ Best HU Model based on Accuracy:")
        print(best_hu)

    print("\n===== SUMMARY: DISEASE MODELS =====")
    print(d_df)

    if not d_df.empty:
        best_d = d_df.loc[d_df['Accuracy'].idxmax()]
        print("\n⭐ Best Disease Model based on Accuracy:")
        print(best_d)

except Exception as e:
    print("\n(Pandas not installed or error during CSV/summary):", e)

print("\n✅ Evaluation finished.")
