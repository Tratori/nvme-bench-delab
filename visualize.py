import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json

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
ENGINES = ["libaio", "io_uring"]
HANDLES_ENGINE_LABELS = [
    mpatches.Patch(color=color, label=label) for label, color in COLORS_ENGINE.items()
]


def import_benchmark():
    with open("benchmark.json", "r") as json_file:
        benchmark = json.load(json_file)
    return benchmark


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
    plt.savefig("read_throughput.png", dpi=400)
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


def main():
    benchmark = import_benchmark()
    visualize_random_read_scalability(benchmark)
    visualize_ssds_vs_reported(benchmark)


if __name__ == "__main__":
    main()
