"""
This module is a utility script for testing the
demonstration LoRAX API referenced in this repository.

Note: This script was generated with ChatGPT.
"""
import requests
import concurrent.futures
import time
import random
from collections import Counter, defaultdict

# Constants
URL = "http://127.0.0.1:8080/generate"
NUM_REQUESTS = 20

prompt = (
    "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half"
    "as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
)

# Model configurations
BASE_MODEL_PARAMS = {
    "inputs": prompt,
    "parameters": {"max_new_tokens": 64}
}
ADAPTER_MODEL_PARAMS = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 64,
        "adapter_id": "vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k"
    }
}

# Initialize response time tracking
response_times = defaultdict(list)
request_counts = Counter()


def send_request(model_type):
    """Send a request to the endpoint and record response time."""
    # Choose payload based on model type
    data = BASE_MODEL_PARAMS if model_type == "base" else ADAPTER_MODEL_PARAMS
    headers = {"Content-Type": "application/json"}

    # Measure request time
    start_time = time.time()
    response = requests.post(URL, json=data, headers=headers)
    response_time = time.time() - start_time

    print(response.content)

    # Store results
    response_times[model_type].append(response_time)
    request_counts[model_type] += 1

    return response.status_code, response_time


def main():
    # Use ThreadPoolExecutor to maximize concurrency
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare tasks with random model selection for each request
        tasks = [
            executor.submit(send_request, random.choice(["base", "adapter"]))
            for _ in range(NUM_REQUESTS)
        ]

        # Wait for all requests to complete
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


# Run the stress test
if __name__ == "__main__":
    main()

# results on my machine
"""
--- Stress Test Results ---
Total requests sent: 20
Base model requests: 10
Adapter model requests: 10

Base Model:
  Average response time: 13.05 seconds
  Max response time: 13.05 seconds
  Min response time: 13.04 seconds

Adapter Model:
  Average response time: 11.30 seconds
  Max response time: 11.30 seconds
  Min response time: 11.30 seconds

Process finished with exit code 0
"""
