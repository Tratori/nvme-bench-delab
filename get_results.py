import os
import sys
import json

# usage e.g.: python get_results.py results/mixed_read_write_results/ results.json
configs = ["nx01", "gx05", "nx02", "gx03", "gx04", "gx01_raid1"]


def main():
    input_dir = sys.argv[1]
    output_path = sys.argv[2]

    results = {}
    for config in configs:
        benchmark_file = input_dir + f"{config}/benchmark"
        if os.path.exists(benchmark_file):
            print(f"Reading file for {config}")
            with open(benchmark_file, "r") as fp:
                benchmark = json.load(fp)
                for key, value in benchmark.items():
                    results[key] = value
    with open(output_path, "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    main()
