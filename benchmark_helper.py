import numpy as np

DEFAULT_AGGS = ["iops"]


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
