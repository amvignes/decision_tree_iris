# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Chargement du jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Évaluation du modèle
score = clf.score(X_test, y_test)
print(f"Précision du modèle: {score:.2f}")

# Visualisation de l'arbre de décision
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()