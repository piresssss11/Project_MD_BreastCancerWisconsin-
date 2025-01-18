import pandas as pd  # Adicionando a importação do pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Carregando o dataset Breast Cancer Wisconsin (Diagnostic) do scikit-learn
data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target


# 1. Visualização da correlação entre as variáveis
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()

# 2. Histogramas das variáveis
df.hist(figsize=(16, 12), bins=20, color='teal')
plt.tight_layout()
plt.show()

# 3. Avaliação de outliers
# Exemplo: Boxplot para algumas variáveis (modifique conforme necessário)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :10], orient='h', palette='viridis')
plt.title("Boxplots das Primeiras 10 Variáveis")
plt.show()

# 4. Normalização das variáveis numéricas
features = data.feature_names
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# 5. Divisão do dataset em treino e teste (80/20)
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Confirmando as dimensões dos conjuntos de dados
print("Dimensões do conjunto de treino (X):", X_train.shape)
print("Dimensões do conjunto de teste (X):", X_test.shape)
print("Dimensões do conjunto de treino (y):", y_train.shape)
print("Dimensões do conjunto de teste (y):", y_test.shape)
