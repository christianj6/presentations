"""
This module is a utility script for testing the
demonstration FastAPI included in this repository.

Note: This script was generated with ChatGPT.
"""
import requests
import concurrent.futures
import time
import random
from collections import Counter, defaultdict

# Constants
URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 20

prompt = (
    "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half"
    " as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
)

# Model configurations
BASE_MODEL_PARAMS = {
    "prompt": prompt,
    "adapter_name": None  # No adapter for base model
}
ADAPTER_MODEL_PARAMS = {
    "prompt": prompt,
    "adapter_name": "one"  # Specific adapter name
}

# Initialize response time tracking
response_times = defaultdict(list)
request_counts = Counter()


def send_request(model_type):
    """Send a request to the FastAPI endpoint and record response time."""
    # Choose payload based on model type
    data = BASE_MODEL_PARAMS if model_type == "base" else ADAPTER_MODEL_PARAMS
    start_time = time.time()

    try:
        response = requests.post(URL, json=data)
        print(response.content)
        response.raise_for_status()  # Raise an error for bad responses
        response_time = time.time() - start_time

        # Store results
        response_times[model_type].append(response_time)
        request_counts[model_type] += 1

        return response.status_code, response_time
    except requests.RequestException as e:
        response_time = time.time() - start_time
        response_times[model_type].append(response_time)
        request_counts[model_type] += 1
        return "Error: {}".format(e), response_time


def stress_test():
    """Conduct a stress test by sending requests to the FastAPI server."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare tasks with random model selection for each request
        tasks = [
            executor.submit(send_request, random.choice(["base", "adapter"]))
            for _ in range(NUM_REQUESTS)
        ]

        # Wait for all requests to complete and gather results
        for future in concurrent.futures.as_completed(tasks):
            status_code, response_time = future.result()

    # Print statistics
    print("\n--- Stress Test Results ---")

    # Throughput and request counts
    print(f"Total requests sent: {NUM_REQUESTS}")
    print(f"Base model requests: {request_counts['base']}")
    print(f"Adapter model requests: {request_counts['adapter']}")

    # Response time stats for base model
    if response_times['base']:
        base_times = response_times['base']
        print("\nBase Model:")
        print(f"  Average response time: {sum(base_times) / len(base_times):.2f} seconds")
        print(f"  Max response time: {max(base_times):.2f} seconds")
        print(f"  Min response time: {min(base_times):.2f} seconds")

    # Response time stats for adapter model
    if response_times['adapter']:
        adapter_times = response_times['adapter']
        print("\nAdapter Model:")
        print(f"  Average response time: {sum(adapter_times) / len(adapter_times):.2f} seconds")
        print(f"  Max response time: {max(adapter_times):.2f} seconds")
        print(f"  Min response time: {min(adapter_times):.2f} seconds")


if __name__ == "__main__":
    stress_test()

# results on my machine
"""
--- Stress Test Results ---
Total requests sent: 20
Base model requests: 8
Adapter model requests: 12

Base Model:
  Average response time: 33.55 seconds
  Max response time: 56.86 seconds
  Min response time: 3.56 seconds

Adapter Model:
  Average response time: 32.83 seconds
  Max response time: 59.64 seconds
  Min response time: 15.81 seconds

Process finished with exit code 0
"""
