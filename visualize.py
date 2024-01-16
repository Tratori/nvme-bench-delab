import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os 
import json
from pathlib import Path
current_directory = Path(__file__).resolve().parent

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


def import_benchmark(file = "benchmark.json"):
    with open(file, "r") as json_file:
        benchmark = json.load(json_file)
    return benchmark

def import_benchmarks(benchmark):
    benchmarks = {}
    path = current_directory / Path("results") / Path(benchmark)
    for root, dirs, files in os.walk(path):
        for file in files:
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
    plt.savefig("mixed_read_write.png", dpi=400)
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
    plt.savefig("reported_read_measured.png", dpi=400)
    plt.show()


def visualize_mixed_read_write(benchmarks): 
    plt.figure(figsize=(12, 8))
    
    machine_configs = list(benchmarks.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns + 1, num_columns, idx)
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
                print(engine, throughputs)
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
    plt.savefig("mixed_read_write.png", dpi=400)
    plt.show()



def main():
    benchmark = import_benchmark()
    visualize_random_read_scalability(benchmark)
    visualize_ssds_vs_reported(benchmark)

    visualize_mixed_read_write(import_benchmarks("mixed_read_write_results"))

if __name__ == "__main__":
    main()
