import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import math
import numpy as np
from pathlib import Path
import seaborn as sns
import pandas as pd
from benchmark_helper import import_logs, aggregate_repeated_benchmark, import_benchmarks, import_benchmark, import_latency_dump
import matplotlib.lines as mlines
sns.set()

COLORS = [
    "blue",
    "green",
    "purple",
    "red",
    "cyan",
    "gray",
    "orange",
    "gold",
    "pink",
    "plum",
    # "sky blue",
    # "reddish purple",
     "black", "brown"

]
COLORS_ENGINE = {
    "libaio": "blue",
    "io_uring": "green",
    "uring": "green",
    "io": "green",
    "spdk": "purple",

}
REPORTED_READS_M = {
    "INTEL_SSDPE2KE016T8": 0.620,
    "Dell_Ent_NVMe_v2_AGN_RI_U.2_1.92TB": 0.920,
    "Samsung_PM991a": 0.350,
    "Samsung_PM961": 0.250,
}
THREADS = [1, 2, 4, 8, 16]
RW = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ENGINES = ["libaio", "io_uring"]
ENGINES_KORONEIA = ["io_uring"]
POLL_MODE_KORONEIA = ["0", "1"]

HANDLES_ENGINE_LABELS = [
    mpatches.Patch(color=color, label=label) for label, color in COLORS_ENGINE.items()
]

ENGINES_BS = ["io_uring"]
BS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BS_LABEL = ["0.5", "1", "2", "4", "8", "16", "32", "64"]
RW_BS = [0, 0.5, 1.0]


RW_RUNTIMES = [0, 0.5, 1.0]
RUNTIMES = [1, 8, 16, 32, 64, 128]
MACHINE_TO_PAPER_NAME = {
    "nx01": "D1",
    "nx02": "D2",
    "nx05": "D3",
    "koroneia": "D4",
}

def visualize_random_read_scalability(benchmarks, threads=THREADS):
    plt.title("4096B - Random Read - IOP/s")
    plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
    plt.xlabel("Threads", fontdict={"fontsize": 12})
    for ssd, benchmark in benchmarks.items():
        for engine in ENGINES:
            runs = [x for x in benchmark if x["IOENGINE"] == engine]
            throughputs = [
                float(run["iops"])
                for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                if int(run["THREADS"]) in threads
            ]
            if engine == ENGINES[0]:
                plt.text(
                    threads[-1],
                    throughputs[-1],
                    ssd.replace("_", " "),
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color="black",
                )

            plt.plot(
                threads,
                throughputs,
                color=COLORS_ENGINE[engine],
            )
    plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
    plt.ylim([0.0, 1.15])
    plt.xticks(threads)
    plt.savefig("figures/random_read_scalability.png", dpi=400)
    plt.show()


def visualize_random_read_scalability2(repeated_benchmarks, metric="iops"):
    aggregate_repeated_benchmark(repeated_benchmarks)

    machine_configs = list(repeated_benchmarks.keys())  # Convert to list if necessary

    num_machines = len(machine_configs)
    num_columns = 1  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns + 1, num_columns, idx)
        plt.title(f"{machine} - Different Block sizes - RW - {metric}")
        plt.ylabel(metric, fontdict={"fontsize": 12})
        plt.xlabel("Blocksize (B)", fontdict={"fontsize": 12})

        plt.xscale('log')

        for ssd, benchmark in repeated_benchmarks[machine].items():
            for i, poll_mode in enumerate(POLL_MODE_KORONEIA):
                runs = [x for x in benchmark if x["IOENGINE"] == "io_uring" and x["IOUPOLL"] == poll_mode]
                print(len(runs))
                iops = np.asarray([
                    float(run[metric + "_mean"])
                    for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                    if int(run["THREADS"]) in THREADS
                ])
                iops_stds = np.asarray([
                    float(run[metric + "_std"])
                    for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                    if int(run["THREADS"]) in THREADS
                ])
                plt.plot(
                    THREADS,
                    iops,
                    label=f"{poll_mode}",
                    color=COLORS[i],
                )
                plt.fill_between(
                    THREADS,
                    iops - iops_stds,
                    iops + iops_stds,
                    color=COLORS[i],
                    alpha=0.2,
                )

        plt.ylim([0.0, 1.15])
        plt.xticks(THREADS, THREADS)
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


def visualize_mixed_read_write(
    benchmarks, suplabelx="Write Percentage", suplabely="Throughput (M IOP/s)"
):

    machine_configs = list(benchmarks.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    fig, axs = plt.subplots(num_machines // num_columns, num_columns, figsize=(12, 8))
    fig.supxlabel(suplabelx)
    fig.supylabel(suplabely)
    for idx, (machine, ax) in enumerate(zip(machine_configs, axs.flatten()), start=1):
        # plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        # plt.xlabel("Write percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in benchmarks[machine].items():
            ax.set_title(f"{machine} - {ssd}")
            for engine in ENGINES:
                runs = [x for x in benchmark if x["IOENGINE"] == engine]
                throughputs = [
                    float(run["iops"])
                    for run in sorted(runs, key=lambda x: float(x["RW"]))
                    if float(run["RW"]) in RW
                ]
                ax.plot(
                    RW,
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )
        if idx == 4:
            ax.legend(
                handles=HANDLES_ENGINE_LABELS, fontsize=12, bbox_to_anchor=(1, 0.5)
            )

        ax.set_ylim([0.0, max(throughputs) * 1.5])
        ax.set_xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/mixed_read_write.png", dpi=400)
    plt.show()


ylabel = {
    "iops": "IOP/s",
    "throughput_gb": "Bandwidth",
}


def visualize_bs_read_write(repeated_benchmarks, metric="iops", normalize=True):
    aggregate_repeated_benchmark(repeated_benchmarks)

    machine_configs = list(repeated_benchmarks.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = num_machines

    fig, axs = plt.subplots(2, num_columns, figsize=(12, 5), sharey=True)
    # fig.suptitle(f"{metric} for Different Block Sizes")

    fig.supxlabel("Block Size (KiB)")
    metric = "iops"
    for idx, (machine, ax) in enumerate(
        zip(machine_configs, axs.flatten()[:num_machines]), start=1
    ):
        if idx == 1:
            ax.set_ylabel(f"{ylabel[metric]} (%)")

        ax.set_title(f"{MACHINE_TO_PAPER_NAME[machine]}")

        ax.set_xscale("log")

        for ssd, benchmark in repeated_benchmarks[machine].items():
            for engine in ENGINES_BS:
                runs = [x for x in benchmark if x["IOENGINE"] == engine]

                max_iops = max(
                    [
                        float(run[metric + "_mean"])
                        for run in sorted(runs, key=lambda x: int(x["BS"]))
                        if int(run["BS"]) in BS
                    ]
                )
                for i, rw in enumerate(RW_BS):

                    iops = np.asarray(
                        [
                            float(run[metric + "_mean"])
                            for run in sorted(runs, key=lambda x: int(x["BS"]))
                            if int(run["BS"]) in BS and float(run["RW"]) == rw
                        ]
                    )
                    iops_stds = np.asarray(
                        [
                            float(run[metric + "_std"])
                            for run in sorted(runs, key=lambda x: int(x["BS"]))
                            if int(run["BS"]) in BS and float(run["RW"]) == rw
                        ]
                    )
                    if normalize:
                        iops_stds /= max_iops
                        iops /= max_iops
                    ax.plot(
                        BS,
                        iops,
                        label=f"{engine} - RW {rw}" if idx == 1 else "_",
                        color=COLORS[i],
                        marker=RW_MARKER[str(rw)],
                    )
                    ax.fill_between(
                        BS,
                        iops - iops_stds,
                        iops + iops_stds,
                        color=COLORS[i],
                        alpha=0.2,
                    )
        ax.set_xticks(BS, BS_LABEL)
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            labelbottom=False,
        )
    metric = "throughput_gb"
    for idx, (machine, ax) in enumerate(
        zip(machine_configs, axs.flatten()[num_machines:]), start=1
    ):
        if idx == 1:
            ax.set_ylabel(f"{ylabel[metric]} (%)")

        ax.set_xscale("log")

        for ssd, benchmark in repeated_benchmarks[machine].items():
            for engine in ENGINES_BS:
                runs = [x for x in benchmark if x["IOENGINE"] == engine]
                max_iops = max(
                    [
                        float(run[metric + "_mean"])
                        for run in sorted(runs, key=lambda x: int(x["BS"]))
                        if int(run["BS"]) in BS
                    ]
                )
                for i, rw in enumerate(RW_BS):

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

                    if normalize:
                        iops_stds /= max_iops
                        iops /= max_iops
                    ax.plot(
                        BS,
                        iops,
                        label=f"_{engine} - RW {rw}",
                        color=COLORS[i],
                        marker=RW_MARKER[str(rw)],
                    )
                    ax.fill_between(
                        BS,
                        iops - iops_stds,
                        iops + iops_stds,
                        color=COLORS[i],
                        alpha=0.2,
                    )
        ax.set_xticks(BS, BS_LABEL)
    plt.tight_layout()  # Adjust layout to prevent overlap
    fig.subplots_adjust(top=0.85)
    fig.legend(fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))

    plt.savefig("figures/bs_read_write.png", dpi=400)
    plt.savefig("figures/bs_d1_d2_d3.pdf")
    plt.show()


def visualize_bs_read_write_after_pause(benchmarks):
    aggregate_repeated_benchmark(benchmarks)
    machine_configs = list(benchmarks.keys())  # Convert to list if necessary

    for machine in machine_configs:
        plt.title(f"{machine} - Read benchmark waiting time after file creation")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Time waited (s)", fontdict={"fontsize": 12})

        for ssd, benchmark in benchmarks[machine].items():
            sorted_benchmark = sorted(benchmark, key=lambda x: int(x["SLEEP_AFTER_INIT"]))
            for engine in ENGINES:
                runs = [x for x in sorted_benchmark if x["IOENGINE"] == engine and x["RW"] == "0.0"]
                throughputs = [float(run["iops_mean"]) for run in runs]
                stds = np.asarray([float(run["iops_std"]) for run in runs])
                plt.plot(
                    [int(run["SLEEP_AFTER_INIT"]) for run in runs],
                    throughputs,
                    color=COLORS_ENGINE[engine],
                )
                plt.fill_between(
                    [int(run["SLEEP_AFTER_INIT"]) for run in runs],
                    throughputs - stds,
                    throughputs + stds,
                    color=COLORS_ENGINE[engine],
                    alpha=0.2,
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

LABELSIZE=20
TICKSIZE=16
LEGENDSIZE=14

def visualize_different_runtimes(repeated_benchmark, ylim=[0.2, 0.8]):
    plt.figure(figsize=(6, 4))
    aggregate_repeated_benchmark(repeated_benchmark)
    machine_configs = list(repeated_benchmark.keys())
    num_machines = len(machine_configs)
    num_columns = 1  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        # plt.title(f"{machine} - Different Runtimes Mixed RW - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": LABELSIZE})
        plt.xlabel("Benchmark duration (s)", fontdict={"fontsize": LABELSIZE})

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
                plt.plot(RUNTIMES[: len(throughputs)], throughputs, color=COLORS[id], marker=RW_MARKER[str(rw)], label=f"{RW_LABEL[str(rw)]}")
                plt.fill_between(
                    RUNTIMES[: len(throughputs)],
                    throughputs - std,
                    throughputs + std,
                    color=COLORS[id],
                    alpha=0.2,
                )

        # plt.legend(
        #     handles=[
        #         mpatches.Patch(color=COLORS[id], label=f"{rw*100} % write")
        #         for id, rw in enumerate(RW_RUNTIMES)
        #     ],
        #     fontsize=12,
        # )
        plt.ylim(ylim)
        plt.yticks([0.2, 0.4, 0.6, 0.8], fontsize=TICKSIZE)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=3,fancybox=True, shadow=True, fontsize=LEGENDSIZE)
        plt.xscale("log")
        plt.xticks([ 2,  8,  32,  128], [ 2,  8,  32,  128], fontsize=TICKSIZE)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/different_runtime_benchmark.pdf", dpi=400)
    plt.show()


RW_new = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
def visualize_mixed_read_write_new(repeated_benchmarks, titles=[], suptitle=""):
    plt.figure(figsize=(12, 8))

    repeated_benchmark = {}
    for benchmark in repeated_benchmarks:
        aggregate_repeated_benchmark(benchmark)
        for machine, ssds in benchmark.items():
            repeated_benchmark[machine] = ssds

    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = num_machines  # You can adjust the number of columns as needed

    num_rows = 0
    for ssd, benchmark in repeated_benchmark[machine_configs[0]].items():
        num_rows = max(num_rows, len(list(set([run["FILENAME"] for run in benchmark]))))

    if suptitle:
        plt.suptitle(
            suptitle,
            fontsize=16,
        )
    for column_idx, machine in enumerate(machine_configs, start=1):
        print(num_machines // num_columns)

        for ssd, benchmark in repeated_benchmark[machine].items():
            unique_filenames = sorted(list(set([run["FILENAME"] for run in benchmark])))
            for row_idx, unique_filename in enumerate(unique_filenames, start=0):

                plt.subplot(num_rows, num_columns, row_idx * num_columns + column_idx)
                if titles:
                    plt.title(titles[column_idx - 1])
                else:
                    plt.title(
                        f"Koroneia - Single SSD - 4096B Page Size - Mixed Read Writes - IOP/s"
                    )
                plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
                plt.xlabel("Write percentage", fontdict={"fontsize": 12})

                for engine in ENGINES:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine and x["FILENAME"] == unique_filename and x["THREADS"] == "32"]
                    rw = [float(run["RW"]) for run in runs]
                    threads = [int(run["THREADS"]) for run in runs]
                    print(threads)
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw
                        ]
                    )

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw
                        ]
                    )
                    if engine == ENGINES[0]:
                        plt.text(
                            rw[-1],
                            throughputs[-1],
                            ssd.replace("_", " ") + unique_filename,
                            fontsize=12,
                            ha="right",
                            va="bottom",
                            color="black",
                        )
                    plt.plot(
                        rw,
                        throughputs,
                        color=COLORS_ENGINE[engine],
                    )
                    plt.fill_between(
                        rw[: len(throughputs)],
                        throughputs - std,
                        throughputs + std,
                        color=COLORS_ENGINE[engine],
                        alpha=0.2,
                    )

                plt.legend(handles=HANDLES_ENGINE_LABELS, fontsize=12)
                # plt.ylim([0, max(throughputs) * 1.1])
                plt.xticks(rw, rw)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/mixed_read_write_new.png", dpi=400)
    plt.show()


RW_LABEL = {"00": "read", "0":"read", "05": "mixed", "0.5": "mixed", "1.0": "write", "10": "write"}
RW_MARKER = {"00": "o", "0":"o", "05": "v", "0.5":"v", "10": "x", "1.0": "x"}


def visualize_logs(logs, bin_size=1):
    # sort by rw
    machine_configs = {k: v for k, v in sorted(logs.items(), key=lambda item: item[0].split("RW")[1][1:3])}
    # machine_configs = sorted(list(logs.keys())) # Convert to list if necessary
    print(machine_configs)
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed
    fig = plt.figure(figsize=(6, 4))
    for idx, machine in enumerate(machine_configs, start=1):
        # plt.title(f"Average Throughput per Thread")
        plt.ylabel("Throughput (MiB / s)", fontdict={"fontsize": LABELSIZE})
        plt.xlabel("Time (min)", fontdict={"fontsize": LABELSIZE})

        logs[machine]["time"] = logs[machine]["time"] / bin_size
        logs[machine]["time"] = logs[machine]["time"].astype(int)

        threads = logs[machine]["threadid"].unique()
        sorted_threads = sorted(threads, key=lambda x: int(x))

        RW = machine.split("RW")[1][1:3]
        engine = machine.split("IOENGINE")[1].split("_")[1]

        logs[machine]["readMibs"] = logs[machine]["readMibs"]
        logs[machine]["writeMibs"] = logs[machine]["writeMibs"]

        logs[machine]["totalMibs"] = logs[machine]["readMibs"] + logs[machine]["writeMibs"]

        mean_total_throughput = logs[machine].groupby("time")["totalMibs"].mean()
        mean_throughput_read = logs[machine].groupby("time")["readMibs"].mean()
        mean_throughput_write = logs[machine].groupby("time")["writeMibs"].mean()
        time_x = logs[machine]["time"].unique()
        #unbinned_time = list(map(lambda x: x * bin_size, list(logs[machine]["time"].unique())))
        # print(len(time_x))
        # for i in time_x:
        #     i = i * bin_size

        # sum_bytes_written = logs[machine].groupby("time")["writeMibs"].sum()
        # cum_sum_written = logs[machine].groupby("time")["writeMibs"].sum().cumsum()
        # print(cum_sum_written)

        # full_write = 100 * 1024*1024*1024
        # for i in range (100):
        #     n_full_writes = i * full_write
        #     if cum_sum_written[cum_sum_written > n_full_writes].empty:
        #         continue
        #     time_100g_written = cum_sum_written[cum_sum_written > n_full_writes].index[0]

        #     plt.axvline(x=time_100g_written, color="red", linestyle="--")
        # plt.plot(sum_bytes_written, label="Bytes written", color="blue", linewidth=0.5, linestyle="--")
        min_throughput = logs[machine].groupby("time")["readMibs"].min()
        max_throughput = logs[machine].groupby("time")["readMibs"].max()

        if len(sorted_threads) < 4:
            for thread in sorted_threads:
                log_thread = logs[machine][logs[machine]["threadid"] == str(thread)]
                print(log_thread)
                plt.plot(
                    log_thread["time"],
                    log_thread["readMibs"],
                    linewidth=2,
                    label=f"Thread {thread}",
                )
        else:
            # plt.plot( mean_throughput_read, label="", color="green", linewidth=1, linestyle="--")
            # plt.plot( mean_throughput_write, label="", color="red", linewidth=1, linestyle="--")
            if engine == "io":
                engine = "uring"
            markevery = 16 if engine == "libaio" else 8
            plt.plot( mean_total_throughput, label=f"{engine} - {RW_LABEL[RW]}" , marker=RW_MARKER[RW], markevery=markevery, markersize=4,  color=COLORS_ENGINE[engine], linewidth=1)

        plt.ylim([50,  150])

    plt.xticks([0, float(600) / bin_size, float(1200) / bin_size, float(1800) /bin_size], ["0", "10", "20", "30"], fontsize=TICKSIZE)
    plt.yticks([75, 125], fontsize=TICKSIZE)
    # plt.legend(fontsize=8)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #       ncol=3,fancybox=True, shadow=True, fontsize=LEGENDSIZE)

    l1 = mlines.Line2D([], [], color=COLORS_ENGINE["io_uring"], label='io_uring')
    l2 = mlines.Line2D([], [], color=COLORS_ENGINE["libaio"], label='libaio')

    l3 = mlines.Line2D([], [], color='black', marker=RW_MARKER["0"], markersize=10, label='read')
    l4 = mlines.Line2D([], [], color='black', marker=RW_MARKER["0.5"], markersize=10, label='mixed')
    l5 = mlines.Line2D([], [], color='black', marker=RW_MARKER["1.0"], markersize=10, label='write')

    plt.legend(handles=[l1, l2, l3, l4, l5], loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3,fancybox=True, shadow=True, fontsize=LEGENDSIZE)

    plt.tight_layout()  # Adjust layout to prevent overlap
        # plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/average_throughput_per_thread.pdf", dpi=400)
    plt.show()

def visualize_scalability(repeated_benchmark, expected_iops=0, title = "Random Reads", save="figures/scalability.png", custom_colors=[]):
    fig = plt.figure(figsize=(4, 8))
    # plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
    # plt.xlabel("Threads", fontdict={"fontsize": 12})
    plt.suptitle(title, fontsize=12)

    # fig,axs = plt.subplots(2,1, figsize = (10,8), sharex=True)

    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    # plt.title(f"Throughput - 4096B - Random Reads - IOP/s")
    for m_idx, machine in enumerate(machine_configs, start=1):
        # plt.subplot(max(num_machines // num_columns, 1), num_columns, idx)
        # plt.title(f"{machine} - 4096B - Random Reads - IOP/s")
        # plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        # plt.xlabel("Threads", fontdict={"fontsize": 12})
        axes = []
        for r_idx,  (ssd, benchmark) in enumerate(repeated_benchmark[machine].items()):
            unique_filenames = sorted(list(set([run["FILENAME"] for run in benchmark])))
            print("here", unique_filenames)
            unique_engines = list(set([run["IOENGINE"] for run in benchmark]))

            for idx, filename in enumerate(unique_filenames, start=1):
                # plt.subplot(max(len(unique_filenames) // num_columns, 1), num_columns, max(idx // num_columns, 1))
                print(unique_filenames)

                ax = plt.subplot(len(unique_filenames), 1, idx)
                axes.append(ax)

                for e_idx, engine in enumerate(unique_engines):
                    runs = [x for x in benchmark if x["IOENGINE"] == engine and x["FILENAME"] == filename]
                    unique_threads = sorted(list(set([int(run["THREADS"]) for run in runs])))

                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                            if int(run["THREADS"]) in unique_threads
                        ]
                    )

                    poll_mode = runs[0]["IOUPOLL"] if "IOUPOLL" in runs[0] else "0"
                    pt_mode = runs[0]["IOUPT"] if "IOUPT" in runs[0] else "0"

                    MARKER_MODE={
                        "0": {
                            "0": "o",
                            "1": "D"
                        },
                        "1": {
                            "0": "x"
                        }
                    }

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: int(x["THREADS"]))
                            if int(run["THREADS"]) in unique_threads
                        ]
                    )


                    engine_label = f"{engine}"
                    if engine == "io_uring":
                        engine_label += " (poll)" if poll_mode == "1" else ""
                        engine_label += " (passthrough)" if pt_mode == "1" else ""

                    plt.plot(
                        unique_threads,
                        throughputs,
                        color=COLORS_ENGINE[engine],
                        marker=MARKER_MODE[poll_mode][pt_mode],
                        label=engine_label,
                    )
                    plt.fill_between(
                        unique_threads,
                        throughputs - std,
                        throughputs + std,
                        color=COLORS_ENGINE[engine],
                        alpha=0.2,
                    )
                    plt.ylim([0.0, max(throughputs) * 1.5])
                    # plt.xticks(unique_threads, unique_threads)
                    # fig.xscale('log')

                    ax.set_xscale('log')
                    ax.set_xticks(unique_threads, [])
                num_ssds = len((filename.split(";")))
                axhline_label = "expected" if m_idx == 1 else ""
                plt.axhline(y=expected_iops * num_ssds, color='r', linestyle='--',  label=axhline_label)
                plt.title( f"{num_ssds} SSDs")
            # axes[-1].get_shared_x_axes().join(axes[-1], *axes[:-1])
            # for x in axes[:-1]:
            #     x.sharex(axes[-1])
            axes[-1].set_xscale('log')
            axes[-1].set_xticks(unique_threads, unique_threads)
            # axes[-2].set_xticks([])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
          fancybox=True, shadow=True, ncol=1)

    fig.supylabel("Throughput (M IOP/s)")
    plt.xlabel("Threads")
    # plt.ylabel("Throoughput (M IOP/s)")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(save, dpi=400)
    plt.show()


def visualize_mixed_read_write_queue_depths(
    repeated_benchmark,
    queue_depths=[128, 256, 1024],
    rw=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
):
    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = 2  # You can adjust the number of columns as needed

    fig, axs = plt.subplots(num_machines // num_columns, num_columns, figsize=(12, 8))
    fig.supxlabel("Write percentage")
    fig.supylabel("Throughput (M IOP/s)")
    fig.suptitle("Mixed Read Writes - Different IO_DEPTHS - IOP/s")
    aggregate_repeated_benchmark(repeated_benchmark)

    for idx, (machine, ax) in enumerate(zip(machine_configs, axs.flatten()), start=1):

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0
            for engine in ENGINES:
                for queue_depth in queue_depths:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine]
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw
                            and int(run["IO_DEPTH"]) == queue_depth
                        ]
                    )

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw
                            and int(run["IO_DEPTH"]) == queue_depth
                        ]
                    )

                    ax.plot(
                        rw,
                        throughputs,
                        color=COLORS[color_id],
                        label=f"{engine} - {queue_depth}",
                    )
                    ax.fill_between(
                        rw[: len(throughputs)],
                        throughputs - std,
                        throughputs + std,
                        color=COLORS[color_id],
                        alpha=0.2,
                    )
                    color_id += 1

        ax.legend(fontsize=12)
        ax.set_ylim([0.0, max(throughputs) * 1.1])
        ax.set_xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("figures/mixed_read_write_different_queue_depths.png", dpi=400)
    plt.show()


def visualize_zero_vs_random_final(repeated_benchmark, threads=[32], rw=[0.0, 0.5, 1.0], inits=["zero", "random"], engines=["libaio", "io_uring"], title=""):
    fig, axes = plt.subplots(nrows=1, ncols=len(repeated_benchmark), figsize=(12, 4), sharey=True)

    machine_configs = sorted(list(repeated_benchmark.keys()))  # Convert to list if necessary

    for idx, (machine, ax) in enumerate(zip(machine_configs, axes), start=1):
        aggregate_repeated_benchmark(repeated_benchmark)

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0
            bars = []
            legend_keys = []
            
            for enine_idx, engine in enumerate(["io_uring", "libaio"]):
                for init_idx, init in enumerate(inits):
                    runs = [x for x in benchmark if x["IOENGINE"] == engine and x["DD_INIT"] == init]
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )

                    bar_width = 0.2
                    n_bars = len(engines) * len(inits)
                    single_width = 1
                    i = enine_idx * len(engines) + init_idx
                    x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )
                    # Draw a bar for every value of that type
                    for x, y in enumerate(throughputs):
                        bar = ax.bar(x + x_offset, y, yerr=std[x], width=bar_width * single_width, color=COLORS_ENGINE[engine], label=f"{engine} - {init} data", hatch=HATCHES[init_idx], error_kw=dict(ecolor='black', lw=2, capsize=3, capthick=1))
                    legend_keys.append(f"{engine} - {init} data")
                    # Add a handle to the last drawn bar, which we'll need for the legend
                    bars.append(bar[0])

                    color_id += 1
                    ax.set_xticks(range(len(rw)))
                    ax.set_xticklabels(["read", "mixed", "write"], fontsize=TICKSIZE)

                    x_axis = "Write null data" if machine == "1ssds_nullwrite" else "Write random data"
                    ax.set_xlabel(x_axis, fontdict={"fontsize": LABELSIZE})
        if idx == 1:
            ax.set_ylabel("Throughput (M IOP/s)", fontdict={"fontsize": LABELSIZE})
            ax.set_yticks([0.0, 0.5, 1.0], [0, 0.5,  1.0], fontdict={"fontsize": TICKSIZE})
            plt.ylim([0.0, 1.1])
            # ax.y("Throughput (M IOP/s)", fontdict={"fontsize": 16})
        
            # ax.set_title(machine)  # Set title for each subplot

    fig.legend(bars, legend_keys, loc='upper center', bbox_to_anchor=(0.51, 1.05),
               fancybox=True, shadow=True, ncol=4, fontsize=LEGENDSIZE)

    axes[0].set_ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
    # fig.supxlabel("Write percentage")
    # fig.suptitle(f"{title} - Mixed Read Writes for files initialized with random data/zeros - IOP/s", fontsize=16)
    fig.legend(bars, legend_keys,loc='upper center',bbox_to_anchor=(0.5, 1.05),
        fancybox=True, shadow=True, ncol=4)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #     ncol=3,fancybox=True, shadow=True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    save_file = title.replace("/", "_")
    safe = save_file.replace(" ", "_")
    plt.savefig(
        f"./figures/{safe}_zero_vs_random_final.pdf",
        dpi=400,
        bbox_inches='tight'
    )
    plt.show()

HATCHES = ["//", "", "|", "-", "+", "x", "o", "O", ".", "*", " "]
def visualize_zero_vs_random(repeated_benchmark, threads=[32], rw=[0.0, 0.5, 1.0], inits =["zero", "random"], engines=["libaio", "io_uring"], title=""):
    fig = plt.figure(figsize=(12, 4))
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = sorted(list(repeated_benchmark.keys()))  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = num_machines  # You can adjust the number of columns as needed
    axes = []
    for idx, machine in enumerate(machine_configs, start=1):
        ax = plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine}")
        axes.append(ax)
        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0
            bars = []
            legend_keys = []
            for enine_idx, engine in enumerate(["io_uring", "libaio"]):
                for init_idx, init in enumerate(inits):
                    # for RW in rw:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine and x["DD_INIT"] == init]
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )

                    # N = 3
                    # ind = np.arange(N)  # the x locations for the groups
                    # width = 0.27       # the width of the bars

                    bar_width = 0.2
                    n_bars = len(engines) * len(inits)
                    single_width = 1
                    i = enine_idx * len(engines) + init_idx
                    x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )
                    # Draw a bar for every value of that type
                    for x, y in enumerate(throughputs):
                        bar = ax.bar(x + x_offset, y, yerr=std[x], width=bar_width * single_width, color=COLORS[enine_idx], label=f"{engine} - {init} init", hatch=HATCHES[init_idx], error_kw=dict(ecolor='black', lw=2, capsize=3, capthick=1))
                    legend_keys.append(f"{engine} - {init} init")
                    # Add a handle to the last drawn bar, which we'll need for the legend
                    bars.append(bar[0])

                    color_id += 1
                    ax.set_xticks(range(len(rw)), rw)
    axes[0].set_ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
    fig.supxlabel("Write percentage")
    fig.suptitle(f"{title} - Mixed Read Writes for files initialized with random data/zeros - IOP/s", fontsize=16)
    plt.legend(bars, legend_keys,loc='right',bbox_to_anchor=(2.25, 0.5),
        fancybox=True, shadow=True, ncol=1)
    plt.tight_layout()  # Adjust layout to prevent overlap
    save_file = title.replace("/","_")
    safe = save_file.replace(" ", "_")
    plt.savefig(
        f"./figures/{safe}_zero_vs_random.png",
        dpi=400,
    )
    plt.show()


def visualize_filled_ssd(repeated_benchmark, threads=[16], rw=[0.0, 1.0], engines=["libaio", "io_uring"]):
    plt.figure(figsize=(12, 8))
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = sorted(list(repeated_benchmark.keys()))  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = num_machines  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(1, num_columns, idx)
        plt.title(f"{machine}")
        # if titles:
        #     plt.title(titles[idx - 1])
        # else:
        #     plt.title(f"{machine} - 4096B Page Size - Mixed Read Writes - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Filled Percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0

            for engine in ["io_uring", "libaio"]:
                for file_size in ["10G"]:
                    for RW in ["0.0", "1.0"]:
                        runs = [x for x in benchmark if x["FILESIZE"] == file_size and x["IOENGINE"] == engine and x["RW"] == RW]
                        print(runs)
                        for run in runs:
                            # fill_status: /dev/nvme0n1p1                      3.8T  3.3T  294G  92% /mnt/nvme1\n
                            if "fill_status" in run:
                                run["fill_percent"] = run["fill_status"].split(" ")[-2][:-1]
                        throughputs = np.asarray(
                            [
                                float(run["iops_mean"])
                                for run in sorted(runs, key=lambda x: float(x["fill_percent"]))
                            ]
                        )
                        print(throughputs)
                        fill_percent = sorted(list(set([x["fill_percent"] for x in runs])))
                        print(fill_percent)
                        # TODO:
                        fill_percentage = []
                        std = np.asarray(
                            [
                                float(run["iops_std"])
                                for run in sorted(runs, key=lambda x: float(x["fill_percent"]))
                            ]
                        )
                        plt.plot(
                            fill_percent,
                            throughputs,
                            color=COLORS[color_id],
                            label=f"{engine} - {file_size}",
                        )
                        plt.fill_between(
                            fill_percent,
                            throughputs - std,
                            throughputs + std,
                            color=COLORS[color_id],
                            alpha=0.2,
                        )
                        color_id += 1
                        plt.xticks(fill_percent, fill_percent)

        plt.legend(fontsize=12)
        plt.ylim([0.0, 1.5])
        # if y_lims:
        #     plt.ylim(y_lims[idx - 1])
        # else:
        #     plt.ylim([0.0, max(throughputs) * 1.5])
        # plt.xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        f"figures/zero_vs_random_{'_'.join(map(lambda x: str(x), engines))}_{'_'.join(map(lambda x: str(x), threads))}.png",
        dpi=400,
    )
    plt.show()

def visualize_10g_vs_100g(repeated_benchmark, threads=[64], rw=[0.0, 0.5, 1.0], engines=["libaio", "io_uring"]):
    plt.figure(figsize=(12, 8))
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = sorted(list(repeated_benchmark.keys()))  # Convert to list if necessary
    num_machines = len(machine_configs)
    num_columns = num_machines  # You can adjust the number of columns as needed

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine}")
        # if titles:
        #     plt.title(titles[idx - 1])
        # else:
        #     plt.title(f"{machine} - 4096B Page Size - Mixed Read Writes - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Write percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0

            for engine in ["io_uring", "libaio"]:
                for file_size in ["10G", "100G"]:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine and x["FILESIZE"] == file_size]
                    print(runs)
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )
                    print(throughputs)

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                        ]
                    )

                    plt.plot(
                        rw,
                        throughputs,
                        color=COLORS[color_id],
                        label=f"{engine} - {file_size}",
                    )
                    plt.fill_between(
                        rw[: len(throughputs)],
                        throughputs - std,
                        throughputs + std,
                        color=COLORS[color_id],
                        alpha=0.2,
                    )
                    color_id += 1
                    plt.xticks(RW, RW)

        plt.legend(fontsize=12)
        # if y_lims:
        #     plt.ylim(y_lims[idx - 1])
        # else:
        #     plt.ylim([0.0, max(throughputs) * 1.5])
        # plt.xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        f"figures/zero_vs_random_{'_'.join(map(lambda x: str(x), engines))}_{'_'.join(map(lambda x: str(x), threads))}.png",
        dpi=400,
    )
    plt.show()


def visualize_mixed_read_write_threads(
    repeated_benchmark,
    threads=[1, 2, 4, 8, 16, 32],
    rw=[0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    engines=["libaio", "io_uring"],
    titles=[],
    suptitle="",
    y_lims=[],
    num_columns=2,
):
    plt.figure(figsize=(12, 8))
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = list(repeated_benchmark.keys())  # Convert to list if necessary
    num_machines = len(machine_configs)

    if suptitle:
        plt.suptitle(suptitle)

    for idx, machine in enumerate(machine_configs, start=1):
        plt.subplot(num_machines // num_columns, num_columns, idx)
        plt.title(f"{machine}")
        if titles:
            plt.title(titles[idx - 1])
        else:
            plt.title(f"{machine} - 4096B Page Size - Mixed Read Writes - IOP/s")
        plt.ylabel("Throughput (M IOP/s)", fontdict={"fontsize": 12})
        plt.xlabel("Write percentage", fontdict={"fontsize": 12})

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0
            for engine in engines:
                for thread in threads:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine]
                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw and int(run["THREADS"]) == thread
                        ]
                    )

                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw and int(run["THREADS"]) == thread
                        ]
                    )

                    plt.plot(
                        rw,
                        throughputs,
                        color=COLORS[color_id],
                        label=f"{engine} - {thread} threads",
                    )
                    plt.fill_between(
                        rw[: len(throughputs)],
                        throughputs - std,
                        throughputs + std,
                        color=COLORS[color_id],
                        alpha=0.2,
                    )
                    color_id += 1

        plt.legend(fontsize=12)
        if y_lims:
            plt.ylim(y_lims[idx - 1])
        else:
            plt.ylim([0.0, max(throughputs) * 1.5])
        plt.xticks(RW)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        f"figures/mixed_read_write_different_threads_{'_'.join(map(lambda x: str(x), engines))}_{'_'.join(map(lambda x: str(x), threads))}.png",
        dpi=400,
    )
    plt.show()


def visualize_mixed_read_write_threads_polished(
    repeated_benchmark,
    threads=[1, 2, 4, 8, 16, 32],
    rw=[0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    engines=["libaio", "io_uring"],
    titles=[],
    suptitle="",
    y_lims=[],
    num_columns=2,
    suplabel="",
    x_ticks=[],
    share_y=True,
    legend_id=None,
):
    aggregate_repeated_benchmark(repeated_benchmark)

    machine_configs = sorted(
        list(repeated_benchmark.keys())
    )  # Convert to list if necessary
    num_machines = len(machine_configs)

    fig, axs = plt.subplots(
        num_machines // num_columns, num_columns, figsize=(12, 8), sharey=share_y
    )
    if suptitle:
        fig.suptitle(suptitle)
    if suplabel:
        fig.supxlabel(suplabel)
    fig.supylabel("Throughput (M IOP/s)", size=16)

    for idx, (machine, ax) in enumerate(zip(machine_configs, axs.flatten()), start=1):
        ax.set_title(f"{machine}")
        if titles:
            ax.set_title(titles[idx - 1], fontdict={"fontsize": 15})
        else:
            ax.set_title(f"{machine} - 4096B Page Size - Mixed Read Writes - IOP/s")
        if not suplabel:
            ax.set_xlabel("Write percentage", fontdict={"fontsize": 16})

        for ssd, benchmark in repeated_benchmark[machine].items():
            color_id = 0
            for engine in engines:
                for thread in threads:
                    runs = [x for x in benchmark if x["IOENGINE"] == engine]

                    throughputs = np.asarray(
                        [
                            float(run["iops_mean"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw and int(run["THREADS"]) == thread
                        ]
                    )
                    if len(throughputs) == 0:
                        continue
                    std = np.asarray(
                        [
                            float(run["iops_std"])
                            for run in sorted(runs, key=lambda x: float(x["RW"]))
                            if float(run["RW"]) in rw and int(run["THREADS"]) == thread
                        ]
                    )
                    if len(engines) > 1:
                        ax.plot(
                            rw,
                            throughputs,
                            color=COLORS[color_id],
                            label=f"{engine} - {'{:2d}'.format(thread)} Threads",
                        )
                    else:
                        ax.plot(
                            rw,
                            throughputs,
                            color=COLORS[color_id],
                            label=f"{'{:2d}'.format(thread)} Threads",
                        )
                    ax.fill_between(
                        rw[: len(throughputs)],
                        throughputs - std,
                        throughputs + std,
                        color=COLORS[color_id],
                        alpha=0.2,
                    )
                    color_id += 1
        if legend_id:
            if legend_id == idx:
                ax.legend(
                    fontsize=12,
                    loc="center left",
                    bbox_to_anchor=(1.0, 0.5),
                    fancybox=True,
                    shadow=True,
                    ncol=1,
                )
        else:
            ax.legend(fontsize=12)
        if y_lims:
            ax.set_ylim(y_lims)
        else:
            ax.set_ylim([0.0, max(throughputs) * 1.5])
        if x_ticks:
            ax.set_xticks(x_ticks)
        else:
            ax.set_xticks(rw)

    ax.legend(loc='right', bbox_to_anchor=(1.8, 0.5),
          fancybox=True, shadow=True, ncol=1, borderaxespad=-0.5, fontsize=16)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        f"figures/mixed_read_write_different_threads_{'_'.join(map(lambda x: str(x), engines))}_{'_'.join(map(lambda x: str(x), threads))}.png",
        dpi=800,
    )
    plt.show()

def visualize_latency(latency_dump):
    for k,df in latency_dump.items():
        print(df.columns)
        print(df["begin"])
        # Convert timestamps to datetime objects
        df['begin'] = pd.to_datetime(df['begin'], unit='ns')
        df['end'] = pd.to_datetime(df['end'], unit='ns')

        # Calculate latency and rounded timestamp
        df['latency'] = (df['end'] - df['begin']).dt.total_seconds()
        df['rounded_begin'] = df['begin'].dt.round('0.5S')

        # Plot latency over time for each thread
        for thread_id, thread_df in df.groupby('threadid'):
            bins = thread_df['rounded_begin'].unique()
            average_latency_per_bin = [thread_df[thread_df['rounded_begin'] == bin]['latency'].mean() for bin in bins]
            plt.plot(bins, average_latency_per_bin, label=f'Thread {thread_id}')
            # plt.plot(thread_df['rounded_begin'], thread_df['latency'], label=f'Thread {thread_id}')

        # Customize plot
        plt.xlabel('Time')
        plt.ylabel('Latency (seconds)')
        plt.title('Latency Over Time for Each Thread')
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    visualize_bs_read_write(import_benchmarks("blocksize_read"))
    return 0
    visualize_logs(import_logs("hour_long/hour_long_vis"), bin_size=16)

    visualize_different_runtimes(import_benchmarks("different_runtimes_results/relevant"))
    visualize_zero_vs_random_final(import_benchmarks("zero_vs_random_koro_final"), threads=[32])
    visualize_logs(import_logs("hour_long/hour_long_vis"), bin_size=16)
    visualize_different_runtimes(import_benchmarks("different_runtimes_results/relevant"))
    # visualize_scalability(import_benchmarks("io_uring_koroneia/upsala"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")

    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_nullwrites_koro"), threads=[32])
    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_koroneia"), threads=[32])

    # visualize_zero_vs_random(import_benchmarks("notnullwrite_zero_vs_random"), threads=[32])
    # visualize_10g_vs_100g(import_benchmarks("10g_vs_100g_nx05"))
    return 0

    return 0
    # visualize_latency(import_latency_dump("local_latency"))
    # return 0

    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_nullwrites_koro"), threads=[32], title="only zeros write")
    # visualize_mixed_read_write_threads_polished(import_benchmarks("nx05_mixed_read_write"), titles=["Intel P5800x", "Intel P5800x (2)", "Intel P5800x (4)", "Intel P5800x", "Intel P5800x (2)", "Intel P5800x (4)"], suptitle="nx05", suplabel="Write Percentage", engines=["libaio"], num_columns=3, x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # visualize_mixed_read_write_threads_polished(import_benchmarks("raid0_nx05"), titles=["Intel P5800x", "Intel P5800x (2)", "Intel P5800x (4)", "Intel P5800x (8)", "Intel P5800x (8) init once", "Intel P5800x (8) init for each run"], suptitle="nx05", suplabel="Write Percentage", engines=["io_uring"], num_columns=6, x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], threads=[1, 2, 4, 8, 16, 32, 64], y_lims=[0,6])
    # visualize_mixed_read_write_threads_polished(import_benchmarks("raid0_nx05"), titles=["Intel P5800x", "Intel P5800x (2)", "Intel P5800x (4)", "Intel P5800x (8)"], suptitle="nx05", suplabel="Write Percentage", engines=["libaio"], num_columns=4, x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], threads=[1, 2, 4, 8, 16, 32, 64], y_lims=[0,6])

    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_koroneia"), threads=[32], title="random data write")

    # visualize_scalability(import_benchmarks("koroneia_scalability"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("io_uring_koroneia/passthrough"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("io_uring_koroneia/upsala/passthrough"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("io_uring_koroneia/upsala/pollmode"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("io_uring_koroneia/upsala/default"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("io_uring_koroneia/upsala"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")

    # (import_benchmarks("io_uring_koroneia"))

    # visualize_filled_ssd(import_benchmarks("filled_ssd")),
    # threads = [1, 2, 4, 8, 16, 32]
    # for i in range(len(threads)):
    #     visualize_mixed_read_write_threads_polished(
    #         import_benchmarks("mixed_read_write_threads"),
    #         engines=["libaio"],
    #         suptitle="libaio Scalability - Linux 5.4.0 (2019)",
    #         titles=["Intel P4610", "Intel P4610 (4 * Raid 0)"],
    #         threads=threads[: i + 1],
    #         y_lims=[[0, 0.8], [0, 3.0]],
    #         x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #         share_y=False,
    #         suplabel="Write Percentage",
    #         legend_id=2,
    #     )
    #     visualize_mixed_read_write_threads_polished(
    #         import_benchmarks("mixed_read_write_threads"),
    #         engines=["io_uring"],
    #         suptitle="io_uring Scalability - Linux 5.4.0 (2019)",
    #         titles=["Intel P4610", "Intel P4610 (4 * Raid 0)"],
    #         threads=threads[: i + 1],
    #         y_lims=[[0, 0.8], [0, 3.0]],
    #         x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #         share_y=False,
    #         suplabel="Write Percentage",
    #         legend_id=2,
    #     )
    # visualize_bs_read_write(import_benchmarks("blocksize_read"))
    # visualize_mixed_read_write_queue_depths(
    #     import_benchmarks("random_read_write_different_queue_depths")
    # )
    # visualize_mixed_read_write(import_benchmarks("mixed_read_write_results"))
    # visualize_filled_ssd(import_benchmarks("filled_ssd"))
    # visualize_logs(import_logs("hour_long"), bin_size=1)
    # visualize_mixed_read_write_threads_polished(
    #     import_benchmarks("koroneia_mixed_read_write_new"), threads=[16], num_columns=3
    # )
    # visualize_mixed_read_write_threads_polished(
    #     import_benchmarks("nx05_mixed_read_write"),
    #     num_columns=3,
    #     titles=["1 Optane SSD", "2 Optane SSD", "4 Optane SSD"],
    #     suptitle="Scalability Threads - nx05 - IOP/s for Threads",
    #     suplabel="Write Percentage",
    #     x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # )
    # visualize_mixed_read_write_threads_polished(
    #     import_benchmarks("nx05_mixed_read_write"),
    #     engines=["io_uring"],
    #     num_columns=3,
    #     titles=["1 Optane SSD", "2 Optane SSD", "4 Optane SSD"],
    #     suptitle="Scalability Threads - nx05 - io_uring - IOP/s for Threads",
    #     suplabel="Write Percentage",
    #     x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    # )

    # visualize_mixed_read_write_new([import_benchmarks("mixed_read_write_new")])
    # benchmark = import_benchmark()
    # visualize_random_read_scalability(benchmark, [1, 2, 4, 8])
    # visualize_ssds_vs_reported(benchmark)

    # visualize_random_read_scalability2(import_benchmarks("random_reads_koroneia"))
    # visualize_mixed_read_write_new(import_benchmarks("read_write_90_percent_ssd"))

    # visualize_mixed_read_write_new(import_benchmarks("read_write_90_percent_ssd"))
    # visualize_mixed_read_write_new(import_benchmarks("read_write_empty_ssd"))
    # visualize_mixed_read_write_new(
    #     [
    #         import_benchmarks("read_write_empty_ssd"),
    #         import_benchmarks("combined_mixed_read_write"),
    #         import_benchmarks("result_999_filled_ssd_koroneia"),
    #     ],
    #     ["empty SSD", "90% filled", "99% filled"],
    #     "Koroneia - Single SSD - 4096B Page Size - Mixed Read Writes - IOP/s",
    # )

    # visualize_scalability(import_benchmarks("koroneia_scalability"), expected_iops=1.0,title="SSD Scalability - Random Reads - Koroneia", save="figures/koroneia_scalability.png")
    # visualize_scalability(import_benchmarks("leanstore_scalability"), expected_iops=1.5,title="SSD Scalability - Random Reads - nx05", save="figures/nx05_scalability.png")

    # visualize_mixed_read_write_new([import_benchmarks("koroneia_mixed_read_write_new")])

    # visualize_mixed_read_write_new(import_benchmarks("fine_granular_mixed_read_write"))
    # visualize_mixed_read_write_new(import_benchmarks("combined_mixed_read_write"))

    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_koroneia"), title="Koroneia")

    # visualize_zero_vs_random(import_benchmarks("zero_vs_random"), inits=["zero", "urandom"], threads=[16], title="nx01/nx02")

    # visualize_zero_vs_random(import_benchmarks("nullwrite_zero_vs_random"), inits=["zero", "random"], threads=[16], title="nx05 null write")
    # visualize_zero_vs_random(import_benchmarks("notnullwrite_zero_vs_random"), inits=["zero", "random"], threads=[16], title="nx05 not null write")
    # visualize_zero_vs_random(import_benchmarks("zero_vs_random_koroneia"))

    # visualize_zero_vs_random(import_benchmarks("nullwrite_zero_vs_random"), inits=["zero", "random"], threads=[16], title="nx05 null write")
    # visualize_zero_vs_random(import_benchmarks("notnullwrite_zero_vs_random"), inits=["zero", "random"], threads=[16], title="nx05 not null write")

    # visualize_mixed_read_write_new([import_benchmarks("nx05_mixed_read_write")], ["1", "2", "4", "8", "16", "32"])

    # visualize_filled_ssd(import_benchmarks("filled_ssd"))
    # visualize_filled_ssd(import_benchmarks("nx05_filled_ssd_new"))

    visualize_filled_ssd(import_benchmarks("filled_ssd_10g_koroneia"))

    # # visualize_bs_read_write(import_benchmarks("results_block_size"))
    # visualize_bs_read_write_after_pause(import_benchmarks("bench_paused_koroneia"))
    # visualize_additional_write_random_read(import_benchmarks("additional_write"))
    # visualize_mixed_read_write_new(import_benchmarks("mixed_read_write"))

    # visualize_bs_read_write(import_benchmarks("blocksize_read"))
    # visualize_bs_read_write(import_benchmarks("blocksize_read_two_ssd"))
    # visualize_bs_read_write(import_benchmarks("blocksize_read"), metric="throughput_gb")
    # visualize_bs_read_write(import_benchmarks("blocksize_read_two_ssd"), metric="throughput_gb")
    # visualize_logs(import_logs("random_reads_koroneia"))


if __name__ == "__main__":
    main()
