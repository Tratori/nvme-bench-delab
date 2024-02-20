import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files into pandas DataFrames
disk_df = pd.read_csv("disk.csv")
flash_df = pd.read_csv("flash.csv")
memory_df = pd.read_csv("memory.csv")

# Add a 'Type' column to each DataFrame
disk_df["Type"] = "Disk"
flash_df["Type"] = "Flash"
memory_df["Type"] = "Memory"

# Concatenate the DataFrames
all_data = pd.concat([disk_df, flash_df, memory_df])

# Convert 'US$/MB' to 'GB/$'
all_data["GB/$"] = 1024 / all_data["US$/MB"]

# Drop rows with non-finite values in 'Year' column

# Convert 'Year' column to integer for plotting
all_data["date"] = all_data["date"].astype(float)
all_data = all_data[all_data["date"] > 2000]


print(all_data)
# Plot the data
plt.figure(figsize=(10, 6))

for storage_type, data in all_data.groupby("Type"):
    plt.scatter(data["date"], data["GB/$"], label=storage_type, marker="o")

plt.title("GB/$ Over Years")
plt.xlabel("Year")
plt.yscale("log")
plt.ylabel("GB/$")
plt.grid(True)
plt.xticks(range(2000, 2025, 5))
plt.tight_layout()
plt.legend(title="Type")
plt.show()
