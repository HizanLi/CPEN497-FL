
# %%
import pandas as pd
import numpy as np

# %%
filepath1 = './fang_our-defense.csv'
filepath2 = './fang_multi-krum.csv'
file1_column_name = 'our defense'
file2_column_name = 'multi-krum'

random_class = True
column_name = 'validation accuracy'

title = 'FANG Attack Validation Accuracy'
ylabel = 'Validation Accuracy'

# %%

df1 = pd.read_csv(filepath1)
df2 = pd.read_csv(filepath2)

# Select the required columns from df1 and df2
df1_selected = df1[['epoch', column_name]].rename(columns={column_name: file1_column_name})
df2_selected = df2[['epoch', column_name]].rename(columns={column_name: file2_column_name})

# Merge the DataFrames on the 'Epoch' column
df3 = pd.merge(df1_selected, df2_selected, on='epoch')

# %%
import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(df3['epoch'], df3[file1_column_name], label=file1_column_name, color='blue')  # Plot for df1
plt.plot(df3['epoch'], df3[file2_column_name], label=file2_column_name, color='red')  # Plot for df2

if random_class:
    plt.plot(df3['epoch'], np.full(301, 10, dtype=int), label='random class', color='green')  # Plot for df2

plt.title(title)  # Title of the plot
plt.xlabel('Epoch')  # Label for the x-axis
plt.ylabel(ylabel)  # Label for the y-axis
plt.ylim(0, 50)
plt.legend()  # Show legend to identify the lines
plt.grid(True)  # Show grid for better readability
plt.show()

# %%
