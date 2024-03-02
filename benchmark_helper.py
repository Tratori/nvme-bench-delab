import numpy as np
import pandas as pd 
from pathlib import Path
import json
import os

current_directory = Path(__file__).parent

DEFAULT_AGGS = ["iops", "throughput_gb"]


def import_benchmark(file="benchmark.json"):
    with open(file, "r") as json_file:
        benchmark = json.load(json_file)
    return benchmark


def import_latency_dump(benchmark):
    latency_dumps = {}
    for path in (current_directory / Path("results") / Path(benchmark)).rglob('dump.csv'):
        config = path.parent.name
        with open(path, "r") as json_file:
            df = pd.DataFrame(columns=["iodepth", "bs", "io_alignment","threadid","reqId","type", "begin", "submit", "end", "addr", "len"])
            df = pd.read_csv(path,skiprows=range(1, 1000001), on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            latency_dumps[config] = df 
    return latency_dumps


def import_benchmarks(benchmark):
    benchmarks = {}
    path = current_directory / Path("results") / Path(benchmark)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("benchmark"):
                file_path = Path(root) / file
                benchmarks[Path(root).name] = import_benchmark(file_path)
    return benchmarks

def import_logs(benchmark):
    logs = {}
    for path in (current_directory / Path("results") / Path(benchmark)).rglob('repetition_0/'):
        config = path.parent.name
        print(config)
        df = pd.DataFrame(columns=["threadid", "time", "writeMibs", "readMibs"], dtype=float)

        for file in path.rglob('*.csv'):
            temp_df = pd.read_csv(file)
            temp_df["threadid"] = file.stem.split("_")[0]
            df = pd.concat([df, temp_df[["threadid", "time", "writeMibs", "readMibs"]]], ignore_index=True)
        logs[config] = df
    return logs

def aggregate_repeated_benchmark(repeated_benchmark, to_be_aggregated=DEFAULT_AGGS):
    for name, nodes in repeated_benchmark.items():
        for ssd, reports in nodes.items():
            for report in reports:
                aggregations = {k: [] for k in to_be_aggregated}
                for repetition in report["repetitions"]:
                    for agg in to_be_aggregated:
                        aggregations[agg].append(float(repetition[agg]))

                for agg in to_be_aggregated:
                    data = np.asarray(aggregations[agg])
                    report[agg + "_std"] = np.std(data)
                    report[agg + "_mean"] = np.mean(data)
                    report[agg + "_med"] = np.median(data)
