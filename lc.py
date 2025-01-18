from sklearn.model_selection import learning_curve

# Função para gerar curvas de aprendizagem
def plot_learning_curve(estimator, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-Validation Score")
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f"{title.lower().replace(' ', '_')}_learning_curve.png")
    plt.show()

# Gerar curvas de aprendizagem para cada modelo
plot_learning_curve(DecisionTreeClassifier(), "Decision Tree", X, y)
plot_learning_curve(SVC(), "SVM", X, y)
plot_learning_curve(RandomForestClassifier(), "Random Forest", X, y)
