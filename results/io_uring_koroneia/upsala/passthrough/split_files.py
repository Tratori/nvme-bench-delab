# this is just a helper script.

import json 
import os
benchmark = json.load(open("benchmark.json", "r"))
print(benchmark.keys())


number_of_files = {
"/mnt/raid0_1ssds/f0" : "1ssds_raid","/mnt/raid0_2ssds/f0": "2ssds_raid", "/mnt/raid0_4ssds/f0" : "4ssds_raid", "/mnt/nvme1/f" : "1ssd_koro", "/mnt/nvme1/f;/mnt/nvme2/f" : "2ssd_koro", "/mnt/nvme1/f;/mnt/nvme2/f;/mnt/nvme3/f" : "3ssd_koro"
, "/dev/nvme0n1;/dev/nvme2n1;/dev/nvme3n1" : "3ssd_pt", "/dev/nvme0n1;/dev/nvme2n1" : "2ssd_pt", "/dev/nvme0n1" : "1ssd_pt"
}

files = {"/mnt/raid0_1ssds/f0" : [],"/mnt/raid0_2ssds/f0": [], "/mnt/raid0_4ssds/f0" : []} 
# {"/mnt/nvme0n1/f" : [], "/mnt/nvme0n1/f;/mnt/nvme1n1/f" : [], "/mnt/nvme0n1/f;/mnt/nvme1n1/f;/mnt/nvme2n1/f;/mnt/nvme3n1/f" : [] , "/mnt/nvme0n1/f;/mnt/nvme1n1/f;/mnt/nvme2n1/f;/mnt/nvme3n1/f;/mnt/nvme4n1/f;/mnt/nvme5n1/f;/mnt/nvme6n1/f;/mnt/nvme7n1/f" : []}
koro_files = {"/mnt/nvme1/f" : [], "/mnt/nvme1/f;/mnt/nvme2/f" : [], "/mnt/nvme1/f;/mnt/nvme2/f;/mnt/nvme3/f" : []}
koro_files_pt = {"/dev/nvme0n1" : [], "/dev/nvme0n1;/dev/nvme2n1" : [], "/dev/nvme0n1;/dev/nvme2n1;/dev/nvme3n1" : []}
files = koro_files_pt

for file in files.keys():
    for key, value in benchmark.items():
        for run in value:
            print(run["FILENAME"])
            if run["FILENAME"] == file:
                files[file].append(run)

for file, runs in files.items():
    # filename = file.replace("/", "_")
    filename = file.replace("/", "_")

    # number_ssds = len(file.split(";"))
    number_ssds = number_of_files[file]
    dir_name = f"{number_ssds}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    json.dump({file : runs}, open(f"{dir_name}/benchmark.json", "w"))
