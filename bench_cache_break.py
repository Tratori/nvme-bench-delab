import sys
import subprocess
import time

from bench import call_iob, create_benchmark_configurations_from_yaml

GBS = [0, 1, 2, 4, 8, 16, 32, 48]


def main():
    io_files = sys.argv[1]
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    yaml_file = sys.argv[5]
    workload = sys.argv[6]

    results = []
    combinations = create_benchmark_configurations_from_yaml(
        yaml_file, workload, io_files
    )

    for gb_after_bench_file in GBS:
        for file in io_files.split(";"):
            subprocess.run(
                [
                    "dd",
                    "if=/dev/zero",
                    f"of={file}",
                    "bs=4k",
                    "oflag=direct",
                    "iflag=fullblock,count_bytes",
                    "count=1G",
                ],
                check=True,
            )

        if gb_after_bench_file > 0:
            subprocess.run(
                ["./write_to_disk.sh", str(gb_after_bench_file), "4"], check=True
            )

        results = call_iob(
            iob_path,
            result_file,
            ssd,
            combinations,
            results,
            {"GB_written_after_file": gb_after_bench_file},
        )

        for file in io_files.split(";"):
            subprocess.run(["rm", file], check=True)


if __name__ == "__main__":
    main()
