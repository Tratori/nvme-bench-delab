import sys
import os
import subprocess
import re
import json
import yaml
import itertools

from copy import deepcopy


def setup_files(io_files):
    def setup():
        for file in io_files.split(";"):
            subprocess.run(
                [
                    "dd",
                    "if=/dev/zero",
                    f"of={file}",
                    "bs=64k",
                    "oflag=direct",
                    "iflag=fullblock,count_bytes",
                    "count=10G",
                ],
                check=True,
            )

    return setup


def cleanup_files(io_files):
    def cleanup():
        for file in io_files.split(";"):
            subprocess.run(["rm", file], check=True)

    return cleanup


def main():
    io_files = sys.argv[1]
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    yaml_file = sys.argv[5]
    workload = sys.argv[6]
    repetitions = sys.argv[7] if len(sys.argv >= 8) else 8

    combinations = create_benchmark_configurations_from_yaml(
        yaml_file, workload, io_files
    )
    call_iob(
        iob_path,
        result_file,
        ssd,
        combinations,
        repetitions=repetitions,
        setup=setup_files(io_files),
        breakdown=cleanup_files(io_files),
    )


def parse_iob_output(output):
    # one newline and one "fin" line behind last output line
    agg = output.split("\n")[-3]

    numbers = re.findall(r"\d+\.\d+|\d+", agg)
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
    with open(result_file, "w") as json_file:
        json.dump({ssd: results}, json_file)


# Calls Leanstore's iob as described in combinations repetition times.
# If setup or breakdown are given, they are called before/after each repetition.
def call_iob(
    iob_path,
    result_file,
    ssd,
    combinations,
    repetitions=8,
    results=[],
    additional_info={},
    setup=None,
    breakdown=None,
):
    for config in combinations:
        run_result = deepcopy(config)
        run_result = dict(run_result, **additional_info)
        run_result["repetitions"] = []
        for repetition in range(repetitions):
            if setup:
                setup()

            result_iob = subprocess.run(
                f"""{iob_path}""",
                text=True,
                capture_output=True,
                shell=True,
                env=dict(os.environ.copy(), **config),
            )

            if breakdown:
                breakdown()

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


def create_benchmark_configurations_from_yaml(yaml_file, workload, io_files):
    combinations = []
    with open(yaml_file, "r") as file:
        yaml_content = yaml.safe_load(file)
        combinations = create_matrix(yaml_content[workload]["matrix"])
        for comb in combinations:
            for arg in yaml_content[workload]["args"].keys():
                comb[arg] = yaml_content[workload]["args"][arg]
            comb["FILENAME"] = io_files
    return combinations


if __name__ == "__main__":
    main()
