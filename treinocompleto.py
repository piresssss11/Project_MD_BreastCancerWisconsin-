import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_breast_cancer

# Carregar o dataset
data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Normalização dos dados
scaler = StandardScaler()
X = df[data.feature_names]
y = df['target']
X = scaler.fit_transform(X)

# Divisão inicial do conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicação do SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Definição dos modelos com class_weight="balanced"
models = {
    'Decision Tree': DecisionTreeClassifier(class_weight="balanced"),
    'SVM': SVC(probability=True, class_weight="balanced"),
    'Random Forest': RandomForestClassifier(class_weight="balanced")
}

# Hiperparâmetros para o GridSearch
param_grid = {
    'Decision Tree': {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None]
    }
}

# Função para treinar e avaliar os modelos
def train_and_evaluate_model(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    print(f"\nTreinando o modelo: {model_name}")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

    # Métricas gerais
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

    # Métricas por classe
    f1_benigno = f1_score(y_test[y_test == 0], y_pred[y_test == 0], zero_division=0)
    f1_maligno = f1_score(y_test[y_test == 1], y_pred[y_test == 1])

    print(f"F1-Score (Geral): {f1:.4f}")
    print(f"Recall (Geral): {recall:.4f}")
    print(f"Precision (Geral): {precision:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"F1 Benigno: {f1_benigno:.4f}, F1 Maligno: {f1_maligno:.4f}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.show()

    return best_model, f1, recall, precision, auc_roc, f1_benigno, f1_maligno

# Treinamento e avaliação
results = {}

for model_name, model in models.items():
    best_model, f1, recall, precision, auc_roc, f1_benigno, f1_maligno = train_and_evaluate_model(
        model_name, model, param_grid[model_name], X_train, X_test, y_train, y_test
    )
    results[model_name] = {
        'best_model': best_model,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'auc_roc': auc_roc,
        'f1_benigno': f1_benigno,
        'f1_maligno': f1_maligno
    }

# Comparação dos resultados
print("\nComparação dos Modelos:")
for model_name, result in results.items():
    print(f"{model_name}: F1-Score = {result['f1_score']:.4f}, Recall = {result['recall']:.4f}, Precision = {result['precision']:.4f}, \
    AUC-ROC = {result['auc_roc']:.4f}, F1 Benigno = {result['f1_benigno']:.4f}, F1 Maligno = {result['f1_maligno']:.4f}")
