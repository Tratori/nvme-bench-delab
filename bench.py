import sys
import os
import subprocess
import re
import json


def main():
    io_files = sys.argv[1]
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    for file in io_files.split(";"):
        subprocess.run(["truncate", "-s", "10G", file], check=True)
    call_iob(iob_path, io_files, result_file, ssd)
    for file in io_files.split(";"):
        subprocess.run(["rm", file], check=True)


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


def call_iob(iob_path, io_files, result_file, ssd):
    results = []
    for threads in [1, 2, 4, 8]:
        for engine in ["libaio", "io_uring"]:
            config = {
                "RUNTIME": "30",
                "FILESIZE": "1G",
                "IOSIZE": "10G",
                "FILENAME": io_files,
                "IOENGINE": engine,
                "INIT": "yes",
                "THREADS": str(threads),
            }

            result_iob = subprocess.run(
                f"""{iob_path}""",
                text=True,
                capture_output=True,
                shell=True,
                env=dict(os.environ.copy(), **config),
            )
            if result_iob.returncode == 0:
                ret = parse_iob_output(result_iob.stdout)
                results.append(dict(ret, **config))
                save_results(result_file, results, ssd)
            else:
                print(result_iob.returncode)
                print(result_iob.stdout)
                print(result_iob.stderr)


if __name__ == "__main__":
    main()
