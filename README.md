# MLOps : API de prédiction Iris (FastAPI + Docker)

Mini-projet MLOps : un modèle scikit-learn (RandomForest) entraîné sur le dataset Iris, exposé via une API FastAPI, validé avec Pydantic, testé avec pytest et conteneurisé avec Docker.

## Structure du projet

```
mlops-iris-api/
├── app/
│   ├── __init__.py
│   └── main.py            # API FastAPI
├── model/
│   └── model.joblib       # modèle entraîné
├── tests/
│   ├── __init__.py
│   └── test_api.py        # tests pytest
├── postman/               # collection Postman exportée
├── train.py               # script d'entraînement
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Prérequis

- Python 3.14 (ou compatible avec les versions de `requirements.txt`)
- Docker Desktop
- Postman (pour les tests manuels)

## Installation

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entraîner le modèle

```powershell
python train.py
```

Le modèle est sauvegardé dans `model/model.joblib`.

## Lancer l'API en local

```powershell
python -m uvicorn app.main:app --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs

## Endpoints

| Méthode | Route      | Description                          |
|---------|------------|--------------------------------------|
| GET     | `/`        | Message d'accueil                    |
| GET     | `/health`  | État de l'API et du modèle           |
| POST    | `/predict` | Prédiction de l'espèce d'une fleur   |

## Exemple de requête

Entrée :

```json
{
  "sepal_length": 6.7,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.3
}
```

Sortie (200 OK) :

```json
{
  "prediction": "virginica",
  "class_id": 2,
  "probabilities": {"setosa": 0.0, "versicolor": 0.0, "virginica": 1.0}
}
```

## Validation d'entrée

Les 4 champs sont obligatoires, de type nombre, strictement supérieurs à 0 et inférieurs ou égaux à 20. Sinon l'API répond **422 Unprocessable Content** avec le détail de l'erreur.

## Tests automatisés

```powershell
python -m pytest -v
```

8 tests : accueil, santé, deux prédictions, champ manquant, valeur négative, mauvais type, valeur trop grande.

## Docker

Construire l'image :

```powershell
docker build -t iris-api:1.0 .
```

Lancer le conteneur :

```powershell
docker run -d --name iris-api -p 8000:8000 iris-api:1.0
```

Vérifier :

```powershell
docker ps
Invoke-RestMethod http://127.0.0.1:8000/health
```

Arrêter et supprimer :

```powershell
docker rm -f iris-api
```

## Tests Postman

La collection exportée se trouve dans le dossier `postman/`. Importer le fichier dans Postman, lancer l'API (locale ou Docker), puis envoyer la requête **Predict Iris**.

## Démo en ligne

API déployée sur SnapDeploy : https://mlops-iris-api-fe956.containers.snapdeploy.app/docs

(Offre gratuite : après environ 15 minutes sans visite, le conteneur s'endort et la première requête peut prendre environ une minute.)

## Technologies

scikit-learn, FastAPI, Pydantic, Uvicorn, pytest, Docker, Postman.