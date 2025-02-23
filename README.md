# Benchmark Comparison Tool
This Python tool compares performance results from two Google Benchmark JSON files, 
analyzing differences in CPU time across benchmarks and visualizing the results in 
the terminal. It provides detailed comparisons, categorizes performance changes, 
and generates plots for both 0D and 1D (Ranges) benchmark data.

# Features
- JSON Parsing: Reads Google Benchmark JSON files and extracts key metrics (real time, CPU time, iterations).
- Benchmark Comparison: Merges two benchmark datasets and computes absolute and relative differences in CPU time.
- Categorization: Classifies performance changes as "superior," "inferior," or "equal" based on a customizable threshold.
- Terminal Visualization: Displays results with color-coded output and generates plots using termplotlib and plotext.
- Size Extraction: Automatically extracts problem sizes from benchmark names for 2D analysis.

# Requirements
- Python 3.x
- `pandas`
- `numpy`
- `matplotlib`
- `termplotlib`
- `plotext`

Install the dependencies with:
```
pip install pandas numpy matplotlib termplotlib plotext
```
# Usage
Run the script with two Google Benchmark JSON files as arguments:
```
python main.py <benchmark1.json> <benchmark2.json>
```
The script will:
- Load and parse both JSON files
- Compare benchmarjs and calculate performance differences
- Display a color-coded summary in the terminal
- Optionally generate terminal based plots


# Example of output

```
BLAS1_op_aligned<float, std::plus< float>>                   [inferior] relative diff: 13.68% rel_min: -73.0% rel_max: 286.1% std: 0.27 count: 1914 size: 1-518144
BLAS1_op_raw<float, std::plus< float>>                       [inferior] relative diff: 8.68% rel_min: -65.9% rel_max: 330.1% std: 0.25 count: 1914 size: 1-518144
BLAS1_op_std_vector<float, std::plus< float>>                [inferior] relative diff: 9.08% rel_min: -78.6% rel_max: 269.6% std: 0.26 count: 1914 size: 1-518144
BLAS1_op_xarray<float, std::plus< float>>                    [inferior] relative diff: 19.35% rel_min: -72.1% rel_max: 91.3% std: 0.15 count: 1914 size: 1-518144
BLAS1_op_xtensor<float, std::plus< float>>                   [inferior] relative diff: 8.98% rel_min: -57.0% rel_max: 167.2% std: 0.11 count: 1914 size: 1-518144
BLAS1_op_xtensor_aligned_64<float, std::plus< float>>        [inferior] relative diff: 8.06% rel_min: -57.1% rel_max: 169.7% std: 0.13 count: 1914 size: 1-518144
BLAS1_op_xtensor_eval<float, std::plus< float>>              [equal   ] relative diff: 4.99% rel_min: -66.6% rel_max: 39.1% std: 0.08 count: 1914 size: 1-518144
BLAS1_op_xtensor_explicit<float, std::plus< float>>          [inferior] relative diff: 42.56% rel_min: -19.4% rel_max: 891.8% std: 0.75 count: 1914 size: 1-518144
BLAS1_op_xtensor_explicit_aligned<float, std::plus< float>>  [inferior] relative diff: 29.82% rel_min: -52.0% rel_max: 419.7% std: 0.46 count: 1914 size: 1-518144
BLAS1_op_xtensor_fixed                                       [inferior] relative diff: 5.17% rel_min: -34.0% rel_max: 70.6% std: 0.33 count: 14 size: 1-16384
BLAS1_op_xtensor_fixed_noalias                               [superior] relative diff: -9.61% rel_min: -53.0% rel_max: 6.7% std: 0.19 count: 14 size: 1-16384
```

- # Future improvements
- Add command-line flags for selecting modes
- Support exporting results to files


