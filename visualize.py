import matplotlib.pyplot as plt
import json

COLORS = {
    "libaio": "blue",
    "io_uring": "green",
}


def import_benchmark():
    with open("benchmark.json", "r") as json_file:
        benchmark = json.load(json_file)
    return benchmark


def visualize_random_read_scalability(benchmarks):
    plt.title("4096B - Random Read - IOP/s")
    plt.ylabel("Throughput (Mio IOP/s)")
    plt.xlabel("Threads")
    for ssd, benchmark in benchmarks.items():
        for engine in ["libaio", "io_uring"]:
            runs = [x for x in benchmark if x["IOENGINE"] == engine]
            plt.plot(
                [1, 2, 4, 8], [float(run["iops"]) for run in runs], color=COLORS[engine]
            )
    plt.show()


def main():
    benchmark = import_benchmark()
    visualize_random_read_scalability(benchmark)


if __name__ == "__main__":
    main()
