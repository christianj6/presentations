"""
This module is a utility script for testing of the
demonstration APIs included in this repository.
"""
import requests
import concurrent.futures
import time
from collections import Counter, defaultdict

url = "http://127.0.0.1:8000/predict"

prompt = "TalkMLOps is"
adapters = ["one", "two"]


def send_request(adapter_name):
    data = {"prompt": prompt, "adapter_name": adapter_name}
    start_time = time.time()

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        return adapter_name, response.json().get("generated_text", ""), elapsed_time
    except requests.RequestException as e:
        elapsed_time = time.time() - start_time
        return adapter_name, f"Error: {str(e)}", elapsed_time


def stress_test(num_requests=50):
    results = Counter()
    response_times = defaultdict(list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(num_requests):
            for adapter_name in adapters:
                print(f"Sending request for adapter '{adapter_name}'")
                futures.append(executor.submit(send_request, adapter_name))

        for future in concurrent.futures.as_completed(futures):
            adapter_name, response_text, elapsed_time = future.result()
            results[adapter_name] += 1
            response_times[adapter_name].append(elapsed_time)

    print("\nStress Test Summary:")
    for adapter_name in adapters:
        total_requests = results[adapter_name]
        total_time = sum(response_times[adapter_name])
        avg_time = total_time / total_requests if total_requests > 0 else 0
        throughput = total_requests / total_time if total_time > 0 else 0
        print(f"\nAdapter '{adapter_name}':")
        print(f"  Total requests: {total_requests}")
        print(f"  Average response time: {avg_time:.4f} seconds")
        print(f"  Throughput: {throughput:.2f} requests/second")


if __name__ == "__main__":
    stress_test()
