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

    df = pd.DataFrame(results)
    extract_size(df)
    return df

def extract_size(df):
    size_pattern = r'[/<](\d+)(?:>)?$'
    df["size"] = df["name"].str.extract(size_pattern).astype(float).astype(int)
    df["benchmark_name"] = df["name"].str.replace(size_pattern, '', regex=True)

def read_benchmark(filename):
    json_data = load_json(filename)
    df = parse_benchmark(json_data)
    return df


def merge_benchmarks(df1,df2):
    merged_df = df1.merge(df2, on="name", suffixes=("_old", "_new"))
    return merged_df

def compare_time(merged_df):
    merged_df["absolute_difference"] = merged_df["cpu_time_new"] - merged_df["cpu_time_old"]
    merged_df["relative_difference"] = merged_df["absolute_difference"] / merged_df["cpu_time_old"]


def categorize_dimension(merged_df):
    merged_df['dimension'] = merged_df.groupby('benchmark_name_new')['benchmark_name_new'].transform('count').gt(1).astype(int)


def categorize(merged_df, threshold):
    def categorize(diff):
        if diff < -threshold :
            return "superior"
        elif diff > threshold :
            return "inferior"
        else:
            return "equal"
    
    merged_df["relative_mean"] = merged_df.groupby("benchmark_name_new")["relative_difference"].transform("mean")
    merged_df["relative_min"] = merged_df.groupby("benchmark_name_new")["relative_difference"].transform("min")
    merged_df["relative_max"] = merged_df.groupby("benchmark_name_new")["relative_difference"].transform("max")
    merged_df["relative_deviation"] = merged_df.groupby("benchmark_name_new")["relative_difference"].transform("std")

    merged_df["comparison"] = merged_df["relative_mean"].apply(categorize)


def compare_benchmarks(df1, df2, threshold=0.05):
    merged_df = merge_benchmarks(df1, df2)
    compare_time(merged_df)
    categorize_dimension(merged_df)
    categorize(merged_df, threshold)
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
    df_unique = df.drop_duplicates(subset=["benchmark_name_new"])
    summary = df_unique["comparison"].value_counts().to_dict()
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

        display_plot_terminal(df, benchmark)



def mode(m):
    if (m == "comparison"):
        print("Select mode comparison")
    elif (m == "show"):
        print("Select mode Show")


def main():
    if len(sys.argv) != 3:
        print("Usage : python main.py <benchmark1.json> <benchmark2.json>")
        sys.exit(1)

    # read 2 google benchbmark files
    filename1, filename2 = sys.argv[1], sys.argv[2]
    df1 = read_benchmark(filename1)
    df2 = read_benchmark(filename2)

#    display_comparison(comparison_df)


    comparison_df = compare_benchmarks(df1, df2)

    display_inferior(comparison_df)
    display_bar(comparison_df)


    display_plot_terminal(comparison_df, "BLAS1_op_raw<float, std::plus< float>>")

    get_2D_benchmarks(comparison_df)


    print_all_plots(comparison_df)

if __name__ == "__main__":
    main()


