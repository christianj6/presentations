import requests
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

x_0 = X_test[0:1]
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": x_0.shape,
          "datatype": "FP64",
          "data": x_0.tolist(),
        }
    ]
}

endpoint = "http://localhost:5555/v2/models/<model_name>/versions/v0.1.0/infer"
response = requests.post(endpoint, json=inference_request)

response.json()
