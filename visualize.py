import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import numpy as np
from pathlib import Path
import seaborn as sns

sns.set()
from benchmark_helper import aggregate_repeated_benchmark

current_directory = Path(__file__).resolve().parent

COLORS = ["blue", "green", "purple"]
COLORS_ENGINE = {
    "libaio": "blue",
    "io_uring": "green",
}
REPORTED_READS_M = {
    "INTEL_SSDPE2KE016T8": 0.620,
    "Dell_Ent_NVMe_v2_AGN_RI_U.2_1.92TB": 0.920,
    "Samsung_PM991a": 0.350,
    "Samsung_PM961": 0.250,
}
THREADS = [1, 2, 4, 8]
RW = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ENGINES = ["libaio", "io_uring"]
HANDLES_ENGINE_LABELS = [
    mpatches.Patch(color=color, label=label) for label, color in COLORS_ENGINE.items()
]

ENGINES_BS = ["io_uring"]
BS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RW_BS = [0, 0.5, 1.0]


RW_RUNTIMES = [0, 0.5, 1.0]
RUNTIMES = [1, 8, 16, 32, 64, 128]

def import_benchmark(file="benchmark.json"):
    with open(file, "r") as json_file:
        benchmark = json.load(json_file)
    return benchmark


def import_benchmarks(benchmark):
    benchmarks = {}
    path = current_directory / Path("results") / Path(benchmark)
    for root, dirs, files in os.walk(path):
        for file in files:
            if(file == "benchmark.json"):
                file_path = Path(root) / file
                benchmarks[Path(root).name] = import_benchmark(file_path)
    return benchmarks


def visualize_random_read_scalability(benchmarks):
    plt.title("4096B - Random Read - IOP/s")
    plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
    plt.xlabel("Threads", fontdict={"fontsize": 12})
    for ssd, benchmark in benchmarks.items():
        for engine in ENGINES:
            runs = [x for x in benchmark if x["IOENGINE"] == engine]
            throughputs = [
                float(run["iops"])
                for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                if int(run["THREADS"]) in THREADS
            ]
            if engine == ENGINES[0]:
                plt.text(
                    THREADS[-1],
                    throughputs[-1],
                    ssd.replace("_", " "),
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color="black",
                )

            plt.plot(
                THREADS,
                throughputs,
                color=COLORS_ENGINE[engine],
            )
    plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
    plt.ylim([0.0, 1.15])
    plt.xticks(THREADS)
    plt.savefig("figures/random_read_scalability.png", dpi=400)
    plt.show()


def visualize_ssds_vs_reported(benchmarks):
    ratio = []
    for ssd, benchmark in benchmarks.items():
        best_run = sorted(benchmark, key=lambda x: float(x["iops"]))[-1]
        ssd_ratio = (float(best_run["iops"]) / REPORTED_READS_M[ssd] - 1) * 100
        ratio.append(ssd_ratio)
        plt.text(
            0.002,
            ssd_ratio,
            ssd.replace("_", " "),
            ha="left",
            va="center",
            color="black",
            fontdict={"fontsize": 12},
        )
    plt.title("Measured vs. Reported random reads/s", fontdict={"fontsize": 16})
    plt.ylabel("Difference in % ", fontdict={"fontsize": 14})
    plt.xlim([-0.01, 0.055])
    plt.xticks([])
    plt.scatter([0] * len(ratio), ratio)
    plt.tight_layout()
    plt.savefig("figures/reported_read_measured.png", dpi=400)
    plt.show()


def visualize_mixed_read_write(benchmarks):
    plt.figure(figsize=(12, 8))

    machine_configs = list(benchmarks.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine} - 4096B - Mixed Read Writes - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Write percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in benchmarks[machine].items():
            for engine in ENGINES:
                runs = [x for x in benchmark if x["IOENGINE"] == engine]
                throughputs = [
                    float(run["iops"])
                    for run in sorted(runs, key=lambda x: float(x["RW"]))
                    if float(run["RW"]) in RW
                ]
                if engine == ENGINES[0]:
                    plt.text(
                        RW[-1],
                        throughputs[-1],
                        ssd.replace("_", " "),
                        fontsize=12,
                        ha="right",
                        va="bottom",
                        color="black",
                    )

                plt.plot(
                    RW,
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )

        plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
        plt.ylim([0.0, max(throughputs) * 1.5])
        plt.xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/mixed_read_write.png", dpi=400)
    plt.show()


def visualize_bs_read_write(repeated_benchmarks, metric="iops"):
    aggregate_repeated_benchmark(repeated_benchmarks)
    plt.figure(figsize=(12, 8))

    machine_configs = list(repeated_benchmarks.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns + 1, num_columns, idx)
        plt.title(f"{machine} - Different Block sizes - RW - {metric}")
        plt.ylabel(metric, fontdict={"fontsize": 12})
        plt.xlabel("Blocksize (B)", fontdict={"fontsize": 12})

        plt.xscale('log')

        for ssd, benchmark in repeated_benchmarks[machine].items():
            for engine in ENGINES_BS:
                for i, rw in enumerate(RW_BS):

                    runs = [x for x in benchmark if x["IOENGINE"] == engine]

                    iops = np.asarray([
                        float(run[metric + "_mean"])
                        for run in sorted(runs, key=lambda x: int(x["BS"]))
                        if int(run["BS"]) in BS and float(run["RW"]) == rw
                    ])
                    iops_stds = np.asarray([
                        float(run[metric + "_std"])
                        for run in sorted(runs, key=lambda x: int(x["BS"]))
                        if int(run["BS"]) in BS and float(run["RW"]) == rw
                    ])
                    plt.plot(
                        BS,
                        iops,
                        label=f"{engine} - RW {rw}",
                        color=COLORS[i],
                    )
                    plt.fill_between(
                        BS,
                        iops - iops_stds,
                        iops + iops_stds,
                        color=COLORS[i],
                        alpha=0.2,
                    )

        plt.legend(
            handles=[
                mpatches.Patch(color=COLORS[id], label=f"{rw*100} % write")
                for id, rw in enumerate(RW_RUNTIMES)
            ],
            fontsize=12,
        )
        plt.xticks(BS, BS)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/pagesize_read_write.png", dpi=400)
    plt.show()


def visualize_bs_read_write_after_pause(benchmarks):
    machine_configs = list(benchmarks.keys())  # Convert to list if necessary

    for machine in machine_configs:
        plt.title(f"{machine} - Read benchmark waiting time after file creation")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Time waited (s)", fontdict={"fontsize": 12})

        for ssd, benchmark in benchmarks[machine].items():
            sorted_benchmark = sorted(benchmark, key=lambda x: int(x["n"]))
            for engine in ENGINES:
                runs = [x for x in sorted_benchmark if x["IOENGINE"] == engine]
                throughputs = [float(run["iops"]) for run in runs]

                plt.plot(
                    [int(run["time_alive"]) for run in runs],
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )

        plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
        plt.ylim([min(throughputs) * 0.9, max(throughputs) * 1.1])

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/paused_read.png", dpi=400)
    plt.show()


def visualize_additional_write_random_read(repeated_benchmark):
    aggregate_repeated_benchmark(repeated_benchmark)
    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary

    for machine in machine_configs:
        plt.title(f"{machine} - Random Read Throughput After Additional Writes to SSD")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Additional writes to SSD (GB)", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            sorted_benchmark = sorted(
                benchmark, key=lambda x: int(x["GB_written_after_file"])
            )
            for engine in ENGINES:
                runs = [x for x in sorted_benchmark if x["IOENGINE"] == engine]
                throughputs = np.asarray([float(run["iops_mean"]) for run in runs])
                stds = np.asarray([float(run["iops_std"]) for run in runs])
                print(engine, throughputs)

                plt.plot(
                    [int(run["GB_written_after_file"]) for run in runs],
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )
                plt.fill_between(
                    [int(run["GB_written_after_file"]) for run in runs],
                    throughputs - stds,
                    throughputs + stds,
                    color=COLORS_ENGINE[engine],
                    alpha=0.2,
                )
        plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)

    plt.savefig("figures/additional_write.png", dpi=400)
    plt.show()


def visualize_different_runtimes(repeated_benchmark):
    plt.figure(figsize=(12, 4))
    aggregate_repeated_benchmark(repeated_benchmark)
    machine_configs = list(repeated_benchmark.keys())
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine} - Different Runtimes Mixed RW - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Runtime of benchmark", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            for id, rw in enumerate(RW_RUNTIMES):
                throughputs = np.asarray(
                    [
                        float(run["iops_mean"])
                        for run in sorted(benchmark, key=lambda x: int(x["RUNTIME"]))
                        if float(run["RW"]) == rw
                    ]
                )
                std = np.asarray(
                    [
                        float(run["iops_std"])
                        for run in sorted(benchmark, key=lambda x: int(x["RUNTIME"]))
                        if float(run["RW"]) == rw
                    ]
                )

                # remove when benchmark is fixed
                plt.plot(RUNTIMES[: len(throughputs)], throughputs, color=COLORS[id])
                plt.fill_between(
                    RUNTIMES[: len(throughputs)],
                    throughputs - std,
                    throughputs + std,
                    color=COLORS[id],
                    alpha=0.2,
                )

        plt.legend(
            handles=[
                mpatches.Patch(color=COLORS[id], label=f"{rw*100} % write")
                for id, rw in enumerate(RW_RUNTIMES)
            ],
            fontsize=12,
        )

        plt.xticks(RUNTIMES)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/runtime_benchmarks.png", dpi=400)
    plt.show()

RW_new = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def visualize_mixed_read_write_new(repeated_benchmark):
    plt.figure(figsize=(12, 8))
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine} - 4096B - Mixed Read Writes - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Write percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            for engine in ENGINES:
                runs = [x for x in benchmark if x["IOENGINE"] == engine]
                throughputs = np.asarray(
                    [
                        float(run["iops_mean"])
                        for run in sorted(runs, key=lambda x: float(x["RW"]))
                        if float(run["RW"]) in RW_new
                    ]
                )

                std = np.asarray(
                    [
                        float(run["iops_std"])
                        for run in sorted(runs, key=lambda x: float(x["RW"]))
                        if float(run["RW"]) in RW_new
                    ]
                )
                if engine == ENGINES[0]:
                    plt.text(
                        RW_new[-1],
                        throughputs[-1],
                        ssd.replace("_", " "),
                        fontsize=12,
                        ha="right",
                        va="bottom",
                        color="black",
                    )

                plt.plot(
                    RW_new,
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )
                plt.fill_between(
                    RW_new[: len(throughputs)],
                    throughputs - std,
                    throughputs + std,
                    color=COLORS_ENGINE[engine],
                    alpha=0.2,
                )

        plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
        plt.ylim([0.0, max(throughputs) * 1.5])
        plt.xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/mixed_read_write_new.png", dpi=400)
    plt.show()


def main():
    benchmark = import_benchmark()
    visualize_random_read_scalability(benchmark)
    visualize_ssds_vs_reported(benchmark)

    visualize_mixed_read_write(import_benchmarks("mixed_read_write_results"))
    visualize_bs_read_write(import_benchmarks("results_block_size"))
    visualize_bs_read_write_after_pause(import_benchmarks("paused_read"))
    visualize_additional_write_random_read(import_benchmarks("additional_write"))
    visualize_different_runtimes(import_benchmarks("different_runtimes_results"))
    visualize_mixed_read_write_new(import_benchmarks("mixed_read_write"))


    visualize_bs_read_write(import_benchmarks("blocksize_read"))
    visualize_bs_read_write(import_benchmarks("blocksize_read_two_ssd"))
    visualize_bs_read_write(import_benchmarks("blocksize_read"), metric="throughput_gb")
    visualize_bs_read_write(import_benchmarks("blocksize_read_two_ssd"), metric="throughput_gb")


if __name__ == "__main__":
    main()
