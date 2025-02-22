import json 
import sys
import pandas as pd
import tabulate


def load_json(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
            print(f"Error during the read of the JSON file : {e}")
            sys.exit(1)


def parse_benchmark(json_data):
    if "benchmarks" not in json_data:
        print("This JSON file do not contain any valid Google Benchmark result...")
        sys.exit(1)

    results = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark.get("name", "Unknown")
        time = benchmark.get("real_time", "N/A")
        cpu_time = benchmark.get("cpu_time", "N/A")
        iterations = benchmark.get("iterations", "N/A")
        results.append({
            "name" : name,
            "real_time": time, 
            "cpu_time": cpu_time,
            "iterations": iterations
            })

    return pd.DataFrame(results)




def display_results(results):
    print("Benchmark results : ")

    print(results)


def main():
    if len(sys.argv) != 2:
        print("Usage : python main.py <benchmark.json>")
        sys.exit(1)
    filename = sys.argv[1]
    json_data = load_json(filename)
    results = parse_benchmark(json_data)
    display_results(results)

if __name__ == "__main__":
    main()


