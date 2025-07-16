# Placeholder content for student_analysis_ml.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy'),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier()
}

# Train-test splits
splits = {'60-40': 0.4, '70-30': 0.3, '80-20': 0.2}

for label, test_size in splits.items():
    print(f"\n📊 Train-Test Split: {label}")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel: {name}")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
print("\n🔁 10-Fold Cross Validation Results:")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=10)
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")

# Plotting Decision Tree
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_scaled, y)
plt.figure(figsize=(10, 6))
plot_tree(dtree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree (Entropy Split)")
plt.savefig("decision_tree_plot.png")
plt.show()
