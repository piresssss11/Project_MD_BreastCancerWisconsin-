from sklearn.model_selection import cross_val_score
import numpy as np

# Função de Validação Cruzada
def cross_validate_model(model, X, y):
    print(f"\nRealizando Validação Cruzada para {model.__class__.__name__}")
    
    # Realiza a validação cruzada usando F1-Score como métrica
    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    
    # Exibindo os resultados
    print(f"F1-Scores por Dobra: {f1_scores}")
    print(f"F1-Score Médio: {np.mean(f1_scores):.4f}")
    
    return np.mean(f1_scores)

# Validação Cruzada para cada modelo
for model_name, model in models.items():
    mean_f1 = cross_validate_model(model, X, y)
    results[model_name]['cross_val_f1'] = mean_f1

# Comparação dos Resultados com Validação Cruzada
print("\nComparação dos Modelos com Validação Cruzada:")
for model_name, result in results.items():
    print(f"{model_name}: F1-Score (Validação Cruzada) = {result['cross_val_f1']:.4f}")
