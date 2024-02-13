# this is just a helper script.

import json 
import os
benchmark = json.load(open("benchmark.json", "r"))
print(benchmark.keys())

files = {"/mnt/nvme0n1/f" : [], "/mnt/nvme0n1/f;/mnt/nvme1n1/f" : [], "/mnt/nvme0n1/f;/mnt/nvme1n1/f;/mnt/nvme2n1/f;/mnt/nvme3n1/f" : [] , "/mnt/nvme0n1/f;/mnt/nvme1n1/f;/mnt/nvme2n1/f;/mnt/nvme3n1/f;/mnt/nvme4n1/f;/mnt/nvme5n1/f;/mnt/nvme6n1/f;/mnt/nvme7n1/f" : []}
#koro_files = {"/mnt/nvme1/f" : [], "/mnt/nvme1/f;/mnt/nvme2/f" : [], "/mnt/nvme1/f;/mnt/nvme2/f;/mnt/nvme3/f" : []}

#files = koro_files

for file in files.keys():
    for key, value in benchmark.items():
        for run in value:
            print(run["FILENAME"])
            if run["FILENAME"] == file:
                
                files[file].append(run)

for file, runs in files.items():
    filename = file.replace("/", "_")

    number_ssds = len(file.split(";"))
    dir_name = f"{number_ssds}ssds"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    json.dump({file : runs}, open(f"{dir_name}/benchmark.json", "w"))
