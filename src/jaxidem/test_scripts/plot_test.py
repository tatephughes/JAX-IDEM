import matplotlib.pyplot as plt
import pandas as pd
import os

print(f"Current working directory: {os.getcwd()}")
# Load the CSV file
df = pd.read_csv("timing_results_cpu_1.csv")

# Assuming the first column is 'Dimension' and the rest are time results
dimensions = df.iloc[:, 0]
#time_results = df.iloc[:, 1:]
time_results = df.iloc[:, 5:]
legend_names = ["Kalman Filter", "Square-root Filter", "Information Filter", "SQRT Inf Filter"]

# Plot each implementation's time results
plt.figure(figsize=(10, 6))
for col, name in zip(time_results.columns, legend_names):
    print(col,name)
    print(time_results[col])
    plt.plot(dimensions, time_results[col], marker='o', label=name)

# Labels and title
plt.xlabel("Observations")
plt.ylabel("Time (ms)")
plt.title("Comparison of filters")

plt.legend()
plt.grid(True)

# Show the plot
plt.show()
