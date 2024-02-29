import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
from matplotlib.ticker import LogFormatterExponent

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

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
all_data["GB/$"] = 1 / all_data["US$/MB"] / 1000
print(all_data["GB/$"])
# Drop rows with non-finite values in 'Year' column

# Convert 'Year' column to integer for plotting
all_data["date"] = all_data["date"].astype(float)
all_data = all_data[all_data["date"] > 2000]

all_data.dropna(subset=["GB/$"], inplace=True)

print(all_data)
# Plot the data
plt.figure(figsize=(10, 6))

for storage_type, data in all_data.groupby("Type"):
    plt.scatter(data["date"], data["GB/$"], label=storage_type, marker="o")

    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(data[["date"]], np.log(data["GB/$"]))
    x_values = data[["date"]].sort_values(by="date")
    y_values = np.exp(model.predict(x_values))

    plt.plot(
        x_values,
        y_values,
        linewidth=3,
    )


plt.xlabel("Year", fontsize=16)
plt.yscale("log")
plt.ylabel("GB/$", fontsize=16)
plt.grid(True)
plt.xticks(list(range(2000, 2025, 5)) + [2024])


def custom_format(x, _):
    if x == 0.001:
        return "0.001"
    elif x == 0.01:
        return "0.01"
    elif x == 0.1:
        return "0.1"
    elif x == 1:
        return "1"
    elif x == 10:
        return "10"
    elif x == 100:
        return "100"
    else:
        return str(x)  # fallback to default formatting for other values


plt.gca().yaxis.set_major_formatter(custom_format)
# plt.ticklabel_format(axis="y", style="plain")  # scilimits=(0, 0))
plt.tight_layout()
plt.legend(fontsize=16)
plt.savefig("../figures/capacity_per_dollar.png", dpi=300)
plt.savefig("../figures/capacity_per_dollar.pdf")
plt.show()
