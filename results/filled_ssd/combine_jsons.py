import os
import json

json_files = [file for file in os.listdir('.') if file.endswith('.json')]


data = {}
for file in json_files:
    print(file)
    with open(file) as f:

        data[file] = json.load(f)


combined = {"nx05_ssd" : []}

for k,v in data.items(): 
    runs = v["nx05_ssd"]
    for run in runs: 

        run["FILL_PERCENT"] = k
        combined['nx05_ssd'].append(run)

json.dump(combined, open(f"./benchmark.json", "w"))

