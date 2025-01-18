
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Carregando o dataset Breast Cancer Wisconsin (Diagnostic) do scikit-learn
data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Exibindo as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Verificando valores ausentes
missing_values = df.isnull().sum()
print("\nValores ausentes em cada coluna:")
print(missing_values)

# Estatísticas descritivas
descriptive_stats = df.describe()
print("\nEstatísticas descritivas do dataset:")
print(descriptive_stats)

# Verificando a distribuição das classes
class_distribution = df['target'].value_counts()
class_distribution_percentage = df['target'].value_counts(normalize=True) * 100
print("\nDistribuição das classes:")
print(class_distribution)
print("\nDistribuição percentual das classes:")
print(class_distribution_percentage)

# Visualização da distribuição das classes
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title("Distribuição das Classes (0 = Maligno, 1 = Benigno)", fontsize=14)
plt.xlabel("Classes")
plt.ylabel("Número de Amostras")
plt.xticks([0, 1], labels=["Maligno", "Benigno"])
plt.show()
