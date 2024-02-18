# model/train.py
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import dagshub

def train_and_log():
    # Load iris dataset
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Train a simple RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy of the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #mlflow.set_tracking_uri("https://dagshub.com/franciskyalo/Iris_DVC.mlflow")
    dagshub.init(repo_owner="franciskyalo", repo_name="Iris_DVC")

    # Log the model and metrics using MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        # Save the model with MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save the model using joblib for local deployment
        joblib.dump(model, "model/model.pkl")

if __name__ == "__main__":
    train_and_log()