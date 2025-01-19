import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib




# Carregar o dataset
data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Normalização dos dados
scaler = StandardScaler()
X = df[data.feature_names]
y = df['target']
X = scaler.fit_transform(X)

# Divisão do dataset em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definição dos modelos
models = {
    'Decision_Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True),  # SVM com probabilidade para calcular AUC-ROC
    'Random_Forest': RandomForestClassifier()
}

# Definição dos hiperparâmetros para o Grid Search
param_grid = {
    'Decision_Tree': {
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Random_Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None]
    }
}

# Função para Treinar Modelos e Realizar GridSearch
def train_and_evaluate_model(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    print(f"\nTreinando o modelo: {model_name}")
    
    # Iniciar o GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
    
    start_time = time.time()
    
    # Treinar o modelo com GridSearch
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    best_model = grid_search.best_estimator_
    
    # Previsões e Avaliação
    y_pred = best_model.predict(X_test)
    
    # Cálculo do F1-Score, Recall e AUC-ROC
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])  # AUC-ROC para probabilidades
    
    # Tempo de execução
    exec_time = end_time - start_time
    
    # Exibição dos resultados
    print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Tempo de Execução: {exec_time:.4f} segundos")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.show()
    
    return best_model, f1, recall, auc_roc, exec_time

# Treino, Avaliação e Comparação dos Modelos
results = {}

for model_name, model in models.items():
    best_model, f1, recall, auc_roc, exec_time = train_and_evaluate_model(model_name, model, param_grid[model_name], X_train, X_test, y_train, y_test)
    results[model_name] = {
        'best_model': best_model,
        'f1_score': f1,
        'recall': recall,
        'auc_roc': auc_roc,
        'execution_time': exec_time
    }

# Comparação dos Resultados
print("\nComparação dos Modelos:")
for model_name, result in results.items():
    print(f"{model_name}: F1-Score = {result['f1_score']:.4f}, Recall = {result['recall']:.4f}, AUC-ROC = {result['auc_roc']:.4f}, Tempo de Execução = {result['execution_time']:.4f} segundos")

# Validação Cruzada com Recall
def cross_validate_model_with_recall(model, X, y):
    print(f"\nRealizando Validação Cruzada para {model.__class__.__name__}")
    
    # Realiza a validação cruzada usando Recall como métrica
    recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall')
    
    # Exibindo os resultados
    print(f"Recall por Dobra: {recall_scores}")
    print(f"Recall Médio: {np.mean(recall_scores):.4f}")
    
    return np.mean(recall_scores)

# Validação Cruzada para cada modelo
for model_name, model in models.items():
    mean_recall = cross_validate_model_with_recall(model, X, y)
    results[model_name]['cross_val_recall'] = mean_recall

# Comparação dos Resultados com Validação Cruzada
print("\nComparação dos Modelos com Validação Cruzada (Recall):")
for model_name, result in results.items():
    print(f"{model_name}: Recall (Validação Cruzada) = {result['cross_val_recall']:.4f}")


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 7))
    
    for model_name, model in models.items():
        # Obter probabilidades preditas
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # Calcular a curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plotar a curva
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curvas ROC")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

plot_roc_curves({name: result['best_model'] for name, result in results.items()}, X_test, y_test)

def plot_learning_curve(model, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(10, 7))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Pontuação de Treino", color="blue")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Pontuação de Validação", color="orange")
    
    plt.title(title)
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Score F1")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Gerar curvas de aprendizagem para cada modelo
for model_name, result in results.items():
    plot_learning_curve(result['best_model'], f"Learning Curve - {model_name}", X, y)

def plot_combined_roc_curves(results, X_test, y_test):
    plt.figure(figsize=(10, 7))
    
    for model_name, result in results.items():
        model = result['best_model']
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("combined_roc_curves.png")  # Salvar como imagem
    plt.show()

# Gerar gráfico com as 3 curvas ROC
plot_combined_roc_curves(results, X_test, y_test)

def plot_combined_learning_curves(results, X, y):
    plt.figure(figsize=(10, 7))
    
    for model_name, result in results.items():
        model = result['best_model']
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, scoring='f1', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        
        plt.plot(train_sizes, test_scores_mean, 'o-', label=f"{model_name} - Validação")
        plt.plot(train_sizes, train_scores_mean, '--', label=f"{model_name} - Treino")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Score F1")
    plt.title("Combined Learning Curves")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("combined_learning_curves.png")  # Salvar como imagem
    plt.show()

# Gerar gráfico com as 3 curvas de aprendizagem
plot_combined_learning_curves(results, X, y)

for k in range(1, len(data.feature_names) + 1):
    X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42, stratify=y)
    best_model.fit(X_train, y_train)
    f1 = f1_score(y_test, best_model.predict(X_test))
    print(f"Número de atributos: {k}, F1-Score: {f1:.4f}")

splits = [0.1, 0.2, 0.3, 0.4]
for split in splits:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=y)
    best_model.fit(X_train, y_train)
    print(f"Teste: {split}, F1-Score: {f1_score(y_test, best_model.predict(X_test)):.4f}")


benigno_f1 = f1_score(y_test[y_test == 0], best_model.predict(X_test)[y_test == 0])
maligno_f1 = f1_score(y_test[y_test == 1], best_model.predict(X_test)[y_test == 1])
print(f"F1 Benigno: {benigno_f1:.4f}, F1 Maligno: {maligno_f1:.4f}")


report = pd.DataFrame(results).T
report.to_csv("report.csv", index=True)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), np.array(data.feature_names)[indices])
plt.title("Importância dos Atributos")
plt.show()

for column in df.columns[:-1]:
    sns.kdeplot(df[column], label=column)
plt.legend()
plt.show()


def apply_cost_matrix(y_true, y_pred, cost_matrix):

    cm = confusion_matrix(y_true, y_pred)  # Gera a matriz de confusão
    cost = np.sum(cm * cost_matrix)       # Calcula o custo total
    return cost
# Geração da matriz de custo
cost_matrix = np.array([[0, 1],  
                        [5, 0]]) 

# Obter previsões
best_model = results['Random_Forest']['best_model']  # Exemplo: Escolha o modelo desejado
y_pred = best_model.predict(X_test)  # Previsões no conjunto de teste

# Aplicar a matriz de custo
total_cost = apply_cost_matrix(y_test, y_pred, cost_matrix)
print(f"Custo Total com a Matriz de Custo: {total_cost}")

# Treinar, avaliar e guardar os modelos
for model_name, model in models.items():
    best_model, f1, recall, auc_roc, exec_time = train_and_evaluate_model(
        model_name, model, param_grid[model_name], X_train, X_test, y_train, y_test
    )
    results[model_name] = {
        'best_model': best_model,
        'f1_score': f1,
        'recall': recall,
        'auc_roc': auc_roc,
        'execution_time': exec_time
    }

# Guardar os modelos treinados
joblib.dump(results['Decision_Tree']['best_model'], "decision_tree_model.pkl")
joblib.dump(results['SVM']['best_model'], "svm_model.pkl")
joblib.dump(results['Random_Forest']['best_model'], "random_forest_model.pkl")

print("Modelos treinados e guardados com sucesso!")


decision_tree = joblib.load("decision_tree_model.pkl")
svm = joblib.load("svm_model.pkl")
random_forest = joblib.load("random_forest_model.pkl")

# Dividir os dados novamente para criar um conjunto de teste diferente
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X, y, test_size=0.3, random_state=43, stratify=y
)

# Avaliar Decision Tree
y_pred_tree = decision_tree.predict(X_test_new)
print("Decision Tree Performance:")
print(classification_report(y_test_new, y_pred_tree))
print(f"F1-Score: {f1_score(y_test_new, y_pred_tree):.4f}")
if hasattr(decision_tree, "predict_proba"):
    print(f"AUC-ROC: {roc_auc_score(y_test_new, decision_tree.predict_proba(X_test_new)[:, 1]):.4f}")

# Avaliar SVM
y_pred_svm = svm.predict(X_test_new)
print("\nSVM Performance:")
print(classification_report(y_test_new, y_pred_svm))
print(f"F1-Score: {f1_score(y_test_new, y_pred_svm):.4f}")
if hasattr(svm, "predict_proba"):
    print(f"AUC-ROC: {roc_auc_score(y_test_new, svm.predict_proba(X_test_new)[:, 1]):.4f}")

# Avaliar Random Forest
y_pred_rf = random_forest.predict(X_test_new)
print("\nRandom Forest Performance:")
print(classification_report(y_test_new, y_pred_rf))
print(f"F1-Score: {f1_score(y_test_new, y_pred_rf):.4f}")
if hasattr(random_forest, "predict_proba"):
    print(f"AUC-ROC: {roc_auc_score(y_test_new, random_forest.predict_proba(X_test_new)[:, 1]):.4f}")
