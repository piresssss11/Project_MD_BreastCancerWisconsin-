Project Overview
The goal of this project is to train and evaluate several machine learning models to classify breast cancer as benign or malignant. Models such as Decision Trees, Support Vector Machine (SVM), and Random Forest are trained on the dataset. Evaluation is performed using various metrics, including F1-Score, Recall, AUC-ROC, and Confusion Matrices. Additionally, techniques like Grid Search for hyperparameter tuning and SMOTE for handling class imbalance are applied.

# Installation
To run this project, you'll need to install several Python libraries. You can install them using the following command:

pip install -r requirements.txt
Here's the list of libraries in the requirements.txt:

pandas
scikit-learn
seaborn
matplotlib
imbalanced-learn
joblib

# Data Preprocessing
The dataset is loaded using load_breast_cancer from sklearn. The features are scaled using StandardScaler to normalize the data. The dataset is then split into training and testing sets (80/20 split).

# Models
The following models are trained and evaluated:

Decision Tree Classifier: A tree-based model that splits the data based on feature thresholds.
Support Vector Machine (SVM): A linear classifier with the option to use non-linear kernels.
Random Forest Classifier: An ensemble method that uses multiple decision trees for better accuracy.
Hyperparameter tuning is performed using GridSearchCV for each model.

# Evaluation
Models are evaluated based on the following metrics:

F1-Score: A balanced measure of precision and recall.
Recall: The proportion of true positives correctly identified.
AUC-ROC: A measure of the model's ability to distinguish between classes.
Confusion Matrix: A matrix to visualize performance across all classes.
Additionally, Cross-Validation is performed to ensure robustness in the evaluation.

# Results
The performance of each model is compared based on the metrics mentioned above. The following are some key results:

Decision Tree: Best F1-Score, Recall, and AUC-ROC values were achieved after hyperparameter tuning.
SVM: Showed great results with a non-linear kernel.
Random Forest: Best performance in terms of accuracy but took longer to train.
ROC Curves and Learning Curves were plotted to visualize model performance.

# Model Saving
Once the models were trained and evaluated, they were saved using joblib. You can load these models later for inference as follows:



# Load models
decision_tree = joblib.load("decision_tree_model.pkl")
svm = joblib.load("svm_model.pkl")
random_forest = joblib.load("random_forest_model.pkl")

# Future Work
Experiment with other algorithms like Gradient Boosting or XGBoost.
Implement Ensemble Methods for combining multiple models.
Fine-tune the models further using more advanced techniques like Bayesian Optimization.
