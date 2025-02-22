import json 
import sys
import pandas as pd
import tabulate

import termplotlib as tpl


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
    print(results)



def compare_benchmarks(df1, df2, threshold=0.05):
    merged_df = df1.merge(df2, on="name", suffixes=("_old", "_new"))
    merged_df["absolute_difference"] = merged_df['cpu_time_new'] - merged_df['cpu_time_old']
    merged_df["relative_difference"] = merged_df["absolute_difference"] / merged_df["cpu_time_old"]

    def categorize(diff):
        if diff < -threshold :
            return "superior"
        elif diff > threshold :
            return "inferior"
        else:
            return "equal"

    merged_df["comparison"] = merged_df["relative_difference"].apply(categorize)
    return merged_df

def display_comparison(df):
    print("Benchmark Comparison : ")
    summary = df["comparison"].value_counts().to_dict()
    for category, count in summary.items():
        print(f"{category}: {count}")

def display_inferior(df):
    inferior_df = df[df["comparison"] == "inferior"]
    if inferior_df.empty:
        print("No worse benchmark result")
    else:
        print(f"\033[91m{inferior_df}\033[0m")
        
def display_bar(df):
    fig = tpl.figure()
    summary = df["comparison"].value_counts().to_dict()
    fig.barh(list(summary.values()), list(summary.keys()))
    fig.show()

def main():
    if len(sys.argv) != 3:
        print("Usage : python main.py <benchmark1.json> <benchmark2.json>")
        sys.exit(1)
    filename1, filename2 = sys.argv[1], sys.argv[2]
    json_data1 = load_json(filename1)
    json_data2 = load_json(filename2)
    df1 = parse_benchmark(json_data1)
    df2 = parse_benchmark(json_data2)

    display_results(df1)
    display_results(df2)

    comparison_df = compare_benchmarks(df1, df2)

#    display_comparison(comparison_df)

    display_inferior(comparison_df)
    display_bar(comparison_df)

if __name__ == "__main__":
    main()


