import sys
import subprocess
import time

from bench import call_iob, create_benchmark_configurations_from_yaml


def main():
    io_files = sys.argv[1]
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    yaml_file = sys.argv[5]
    workload = sys.argv[6]
    wait_seconds = int(sys.argv[7])
    repeat_n = int(sys.argv[8])

    for file in io_files.split(";"):
        subprocess.run(
            [
                "dd",
                "if=/dev/zero",
                f"of={file}",
                "bs=4k",
                "iflag=fullblock,count_bytes",
                "count=5G",
            ],
            check=True,
        )
    combinations = create_benchmark_configurations_from_yaml(
        yaml_file, workload, io_files
    )

    results = []
    for i in range(repeat_n):
        results = call_iob(
            iob_path,
            result_file,
            ssd,
            combinations,
            5,
            results,
            {"n": i, "time_alive": i * wait_seconds},
        )
        time.sleep(wait_seconds)

    for file in io_files.split(";"):
        subprocess.run(["rm", file], check=True)


if __name__ == "__main__":
    main()
