from bench import call_iob, create_benchmark_configurations_from_yaml, setup_files, cleanup_files
import sys
from pathlib import Path
import subprocess

from concurrent.futures import ThreadPoolExecutor

def fill_ssd(io_files, ssd_size, fill_percent, idx):
    dir = Path(io_files).parent

    fill_files = [] 
    to_be_filled = int(ssd_size * fill_percent)

    num_10g_files = to_be_filled // 10
    size_mod_file = to_be_filled % 10
    
    for i in range (num_10g_files + 1):
        fill_file = dir / f"{idx}_fill_file_{i}"
        fill_files.append(fill_file)

    def run_dd(filename, size):
        print("filename:", filename, " size: ", size)
        subprocess.run(
            [
                "dd",
                "if=/dev/zero",
                f"of={filename}",
                "oflag=direct",
                "bs=4M",
                "iflag=fullblock,count_bytes",
                f"count={size}G",
            ],
            check=True,
        )
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(run_dd, str(filename), 10) for filename in fill_files[:-1]]
        futures.append(executor.submit(run_dd, fill_files[-1], size_mod_file))
        for future in futures:
            future.result()
    return fill_files 

def main():
    io_files = sys.argv[1] if sys.argv[1] != "None" else None
    iob_path = sys.argv[2]
    result_file = sys.argv[3]
    ssd = sys.argv[4]
    yaml_file = sys.argv[5]
    workload = sys.argv[6]

    ssd_size_gb_without_benchmarking_file = int(sys.argv[7])
    print("SSD-size: ", ssd_size_gb_without_benchmarking_file)
    
    # fill_level = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
    fill_level_steps = [0.1, 0.2, 0.2, 0.2, 0.2, 0.05, 0.04, 0.01]
    assert (sum(fill_level_steps) == 1.0)


    for idx, fill in enumerate(fill_level_steps):
        fill_files = fill_ssd(io_files, ssd_size_gb_without_benchmarking_file, fill, idx)

        get_fill_status_cmd = "df -lH | grep /dev/nvme0n1p1"
        fill_status = subprocess.run(get_fill_status_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        additional_info = {
            "fill_percent": fill,
            "fill_status": fill_status.stdout.decode("utf-8")
        }

        fill_result_file = result_file.replace(".json", f"_fill_{fill}.json")

        combinations = create_benchmark_configurations_from_yaml(
            yaml_file, workload, io_files
        )
        call_iob(
            iob_path,
            fill_result_file,
            ssd,
            combinations,
            repetitions=5,
            setup=setup_files,
            teardown=cleanup_files,
            additional_info=additional_info
        ) 

        # for file in fill_files:
        #     subprocess.run(["rm", file], check=True)

if __name__ == "__main__":
    main()
