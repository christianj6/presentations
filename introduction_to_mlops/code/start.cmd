@echo off
setlocal

rem Define URLs and file paths
set "AIRFLOW_URL=https://airflow.apache.org/docs/apache-airflow/2.10.3/docker-compose.yaml"

set "AIRFLOW_FILE=docker-compose.airflow.yml"
set "MLFLOW_FILE=docker-compose.mlflow.yml"

rem Download Airflow Docker Compose file
echo Downloading Airflow Docker Compose file...
powershell -Command "Invoke-WebRequest -Uri %AIRFLOW_URL% -OutFile %AIRFLOW_FILE%"
if not exist "%AIRFLOW_FILE%" (
    echo Failed to download Airflow file
    exit /b 1
)

rem Run airflow-init using the Airflow Compose file
echo Initializing Airflow with 'docker-compose up airflow-init'...
docker-compose -f "%AIRFLOW_FILE%" up airflow-init
if %errorlevel% neq 0 (
    echo Airflow initialization failed
    exit /b 1
)

rem Bring up both Airflow and MLflow services
echo Starting services for both Airflow and MLflow...
docker-compose -f "%AIRFLOW_FILE%" -f "%MLFLOW_FILE%" up --build -d
if %errorlevel% neq 0 (
    echo Failed to start services
    exit /b 1
)

echo Both Airflow and MLflow services are now up and running!
