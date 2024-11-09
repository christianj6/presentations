#!/bin/bash
set -e

# Define URLs and file paths
AIRFLOW_URL="https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml"
AIRFLOW_FILE="docker-compose.airflow.yml"
MLFLOW_FILE="docker-compose.mlflow.yml"

# Download Airflow Docker Compose file
echo "Downloading Airflow Docker Compose file..."
curl -o "$AIRFLOW_FILE" "$AIRFLOW_URL"
if [ ! -f "$AIRFLOW_FILE" ]; then
    echo "Failed to download Airflow file"
    exit 1
fi

# Run airflow-init using the Airflow Compose file
echo "Initializing Airflow with 'docker-compose up airflow-init'..."
docker-compose -f "$AIRFLOW_FILE" up airflow-init
if [ $? -ne 0 ]; then
    echo "Airflow initialization failed"
    exit 1
fi

# Bring up both Airflow and MLflow services
echo "Starting services for both Airflow and MLflow..."
docker-compose -f "$AIRFLOW_FILE" -f "$MLFLOW_FILE" up --build -d
if [ $? -ne 0 ]; then
    echo "Failed to start services"
    exit 1
fi

echo "Both Airflow and MLflow services are now up and running!"
