from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy sur le jeu de test : {accuracy:.3f}")

    Path("model").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "target_names": list(iris.target_names)},
        "model/model.joblib",
    )
    print("Modèle sauvegardé dans model/model.joblib")


if __name__ == "__main__":
    main()