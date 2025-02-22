import json 
import sys
import pandas as pd
import tabulate

import termplotlib as tpl

import matplotlib.pyplot as plt

import numpy as np 
import plotext as plt2

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

def extract_size(df):
    size_pattern = r'[/<](\d+)(?:>)?$'
    df["size"] = df["name"].str.extract(size_pattern).astype(float).astype(int)
    df["benchmark_name"] = df["name"].str.replace(size_pattern, '', regex=True)


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



def mean_2D(df, benchmark_name):
    result = df[df["benchmark_name_new"] == benchmark_name]
    difference = (result["cpu_time_old"]-result["cpu_time_new"])/result["cpu_time_old"]
    mean = np.mean(difference)
    return mean



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



def display_plot_terminal(comparison_df, name):
    result = comparison_df[comparison_df["benchmark_name_new"] == name]
    print(f"\033[1;4mTestcase : {name}\033[0m")

    linex = [min(result["size_new"]), max(result["size_new"])]

    plt2.clf()

    plt2.subplots(1,2)#
#    plt2.subplot(1,1).title("Comparison")
    plt2.subplot(1,1).plotsize(plt2.tw()//2, plt2.th()//4)
    plt2.subplot(1,1).theme('pro')
    plt2.subplot(1,1).xscale('log')
    plt2.subplot(1,1).yscale('log')
    plt2.subplot(1,1).xlabel('Size')
    plt2.subplot(1,1).ylabel('Time')
    plt2.subplot(1,1).plot(result["size_new"], result["cpu_time_old"], label="old")
    plt2.subplot(1,1).plot(result["size_new"], result["cpu_time_new"], label="new")
#    plt2.subplot(1,1).title(f"Benchmark  {name}")
    plt2.subplot(1,1).yscale('log')
    plt2.subplot(1,1).xscale('log')
    plt2.subplot(1,1).xlabel("Size")
    plt2.subplot(1,1).ylabel("Time")

#    plt2.subplot(1,2).title("difference")
    plt2.subplot(1,2).plot(linex, [0.0,0.0], label='reference')
    plt2.subplot(1,2).plot(result["size_new"], (result["cpu_time_old"]-result["cpu_time_new"])/result["cpu_time_old"], label="difference")
             
    plt2.subplot(1,2).plotsize(plt2.tw()//2, plt2.th()//4)
    plt2.subplot(1,2).theme('pro')
    plt2.subplot(1,2).xscale('log')
    plt2.subplot(1,2).theme("pro")
    plt2.subplot(1,2).xlabel("Size")
    plt2.subplot(1,2).ylabel("Relative time difference")

    plt2.show()


def get_2D_benchmarks(df, threshold=2):
    counts = df["benchmark_name_old"].value_counts()
    above_threshold = counts[counts > threshold].index.tolist()
    print(f"Benchmarks avec plus de {threshold} occurrences : {above_threshold}")

    return above_threshold


def print_all_plots(df):
    above_threshold = get_2D_benchmarks(df, 2)
    for benchmark in above_threshold : 

        mean = mean_2D(df, benchmark)
        print(f"Average gain : {mean:.3f}")
        display_plot_terminal(df, benchmark)



def main():
    if len(sys.argv) != 3:
        print("Usage : python main.py <benchmark1.json> <benchmark2.json>")
        sys.exit(1)
    filename1, filename2 = sys.argv[1], sys.argv[2]
    json_data1 = load_json(filename1)
    json_data2 = load_json(filename2)
    df1 = parse_benchmark(json_data1)
    df2 = parse_benchmark(json_data2)

    extract_size(df1)
    extract_size(df2)

    display_results(df1)
    display_results(df2)

    comparison_df = compare_benchmarks(df1, df2)

#    display_comparison(comparison_df)

    display_inferior(comparison_df)
    display_bar(comparison_df)


    display_plot_terminal(comparison_df, "BLAS1_op_raw<float, std::plus< float>>")

    get_2D_benchmarks(comparison_df)


    print_all_plots(comparison_df)

if __name__ == "__main__":
    main()


