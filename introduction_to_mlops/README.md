### Introduction to MLOps
Machine Learning Operations (MLOps) is an increasingly important discipline as the complexity of ML-based applications increases and the unique demands of model versioning, deployment, and monitoring grow along with specialized technologies.

This directory contains the original presentation slides from the talk, as well as configuration files to support the setup of a local demo MLOps system.

***

#### Getting Started
1. Start [Docker](https://docs.docker.com/get-started/) on your machine.
2. Navigate to the /code directory and run either start.cmd or start.sh, depending on your OS. This should download the necessary docker-compose file and start services locally.
3. Visit the following locations in your browser to observe the services:
   - **mlflow**: http://localhost:5000
   - **airflow**: http://localhost:8080 (user: airflow, pass: airflow)
6. Run the training.py DAG by navigating to the airflow webserver UI in your browser and triggering the DAG. This should result in a new MLflow experiment run with a logged sklearn model.
7. Register the model from the MLflow webserver UI by following the appropriate prompts.
8. Adjust the MLServer model configuration file to reference the new model you registered, then restart the MLServer container.
9. Test the MLServer endpoint using the test.py script.

***

#### Notes and Attribution
- MLflow docker-compose example is taken from [this repository](https://github.com/sachua/mlflow-docker-compose).
- Training DAG code is taken from the [MLServer sklearn example](https://mlserver.readthedocs.io/en/latest/examples/sklearn/README.html).
- **WARNING FOR MAC USERS**: You may experience issues with the MLflow container attempting to expose the allocated port 5000. This may be due to [a setting which you need to adjust](https://stackoverflow.com/questions/72369320/why-always-something-is-running-at-port-5000-on-my-mac), otherwise you need to remap the ports configured in the demo.

***
