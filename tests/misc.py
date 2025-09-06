import time
from dotenv import load_dotenv

def benchmark_load_dotenv(iterations: int = 1000):
    start = time.perf_counter()
    for _ in range(iterations):
        load_dotenv(override=True)  # force reload each time
    end = time.perf_counter()
    elapsed = end - start
    avg = elapsed / iterations
    print(f"Ran load_dotenv {iterations} times")
    print(f"Total time: {elapsed:.6f} seconds")
    print(f"Average per call: {avg * 1000:.6f} ms")

if __name__ == "__main__":
    benchmark_load_dotenv(100)