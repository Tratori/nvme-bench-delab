import sys
import os
import subprocess
import re
import json
import yaml
import itertools

from copy import deepcopy
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

def setup_files(config):
    filenames = config["FILENAME"].split(";")

    def run_dd(filename):
        subprocess.run(
            [
                "dd",
                "if=/dev/zero",
                f"of={filename}",
                "bs=64k",
                "oflag=direct",
                "iflag=fullblock,count_bytes",
                "count=10G",
            ],
            check=True,
        )

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_dd, filename) for filename in filenames]

        for future in futures:
            future.result()

def setup_output_dir(result_file, config_str, repetition):
    path =  Path(result_file).parent / Path(config_str) / Path(repetition)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory {path}")
    return str(path)

def cleanup_files(config):
    for file in config["FILENAME"].split(";"):
        subprocess.run(["rm", file], check=True)


def main():
    io_files = sys.argv[1] if sys.argv[1] != "None" else None
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    yaml_file = sys.argv[5]
    workload = sys.argv[6]

    combinations = create_benchmark_configurations_from_yaml(
        yaml_file, workload, io_files
    )
    call_iob(
        iob_path,
        result_file,
        ssd,
        combinations,
        repetitions=5,
        setup=setup_files,
        teardown=cleanup_files
    )


def parse_iob_output(output):
    # one newline and one "fin" line behind last output line
    agg = output.split("\n")[-3]

    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", agg)
    if len(numbers) != 7:
        print("Agg numbers output not in expected format : ", numbers)
        print(agg)
        sys.exit(2)

    return {
        "iops": numbers[0],
        "readss": numbers[1],
        "writess": numbers[2],
        "throughput_gb": numbers[3],
        "throughput_gb_reads": numbers[4],
        "throughput_gb_writes": numbers[5],
        "max_iops": numbers[6],
    }


def save_results(result_file, results, ssd):
    with open(result_file, "w+") as json_file:
        json.dump({ssd: results}, json_file)


# Calls Leanstore's iob as described in combinations repetition times.
# If setup or teardown are given, they are called before/after each repetition.
def call_iob(
    iob_path,
    result_file,
    ssd,
    combinations,
    repetitions=5,
    results=[],
    additional_info={},
    setup=None,
    teardown=None
):
    for config in combinations:
        run_result = deepcopy(config)
        run_result = dict(run_result, **additional_info)
        run_result["repetitions"] = []
        for repetition in range(repetitions):
            config["OUTPUT_DIR"] = setup_output_dir(result_file, config["CONFIG_STR"], f"repetition_{repetition}/") + "/"
            if setup:
                setup(config)

            result_iob = subprocess.run(
                f"""{iob_path}""",
                text=True,
                capture_output=True,
                shell=True,
                env=dict(os.environ.copy(), **config),
            )

            if teardown:
                teardown(config)

            if result_iob.returncode == 0:
                ret = parse_iob_output(result_iob.stdout)
                ret["repetition"] = repetition
                run_result["repetitions"].append(ret)
            else:
                print(result_iob.returncode)
                print(result_iob.stdout)
                print(result_iob.stderr)

        results.append(run_result)
        save_results(result_file, results, ssd)

    return results


def create_matrix(yaml_content):
    dimensions = yaml_content.keys()
    dimension_values = [yaml_content[dim] for dim in dimensions]
    combinations = list(itertools.product(*dimension_values))
    result = []
    for combo in combinations:
        result.append(dict(zip(dimensions, combo)))
    return result

def directory_name(comb): 
    dir_name =  '__'.join([f'{key}_{value}' for key, value in comb.items()])
    keepcharacters = ('_')
    return "".join(c for c in dir_name if c.isalnum() or c in keepcharacters).rstrip()

def create_benchmark_configurations_from_yaml(yaml_file, workload, io_files):
    combinations = []
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)
        combinations = create_matrix(yaml_content[workload]["matrix"])
        for comb in combinations:
            for arg in yaml_content[workload]["args"].keys():
                comb[arg] = yaml_content[workload]["args"][arg]
            if io_files:
                comb["FILENAME"] = io_files
            assert ("FILENAME" in comb and comb["FILENAME"] is not None)
            comb["CONFIG_STR"] = directory_name(comb)
    return combinations


if __name__ == "__main__":
    main()
