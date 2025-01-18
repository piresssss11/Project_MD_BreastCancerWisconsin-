import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Carregar o dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Verificar distribuição inicial
print(f"Distribuição das classes antes do balanceamento: {Counter(y)}")

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Aplicar SMOTE no conjunto de treino
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Distribuição das classes após SMOTE: {Counter(y_train_balanced)}")

# Escalar os dados para o SVM
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Treinar os modelos
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, class_weight='balanced', random_state=42)
}

results = {}

for model_name, model in models.items():
    if model_name == "SVM":
        model.fit(X_train_balanced_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[model_name] = {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc
    }
    
    print(f"Modelo: {model_name}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Matriz de Confusão:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Plotar curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot da curva ROC
plt.plot([0, 1], [0, 1], 'k--', label="Aleatório")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC")
plt.legend()
plt.show()
