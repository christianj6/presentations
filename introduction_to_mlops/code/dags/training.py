import os
from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.utils.dates import days_ago

# these values are copied from the .env file; in production you
# would want a more sophisticated variable management approach of course
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'model_training_and_upload',
    default_args=default_args,
    description='A simple DAG for training a model and uploading it to MLflow',
    schedule_interval=None,  # Run on manual trigger or trigger interval
)

# Define the Python function for training and uploading the model
def train_and_upload_model():
    # Importing inside function to ensure it runs in the virtual environment
    from sklearn import datasets, svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    # Ensure MLflow is connected to the local server
    mlflow.set_tracking_uri("http://mlflow_server:5000")  # Set the tracking URI to the local MLflow server

    # Load the dataset
    digits = datasets.load_digits()

    # Flatten the images to create a feature matrix
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    # Create and train a Support Vector Classifier (SVC)
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train, y_train)

    # Predict on the test set and evaluate accuracy
    predicted = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print(f"Model accuracy: {accuracy}")

    model_signature = infer_signature(X_train, y_train)

    # Start a new MLflow run and create a new experiment (if not already done)
    with mlflow.start_run():
        # Log parameters and metrics
        mlflow.log_param("model", "SVM")
        mlflow.log_param("gamma", 0.001)
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(classifier, "svm_model", signature=model_signature)

    print("Model logged to MLflow")


# Create the PythonOperator task for training and uploading the model
train_and_upload_task = PythonVirtualenvOperator(
    task_id='train_and_upload_model',
    python_callable=train_and_upload_model,
    requirements=["scikit-learn", "mlflow", "boto3"],
    system_site_packages=False,  # Avoid using system site packages
    dag=dag,
)

train_and_upload_task
