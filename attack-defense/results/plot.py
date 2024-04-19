
# %%
import pandas as pd
import numpy as np

# %%
filepath1 = './baseline.csv'
filepath2 = './baseline-no-attack.csv'
filepath4 = './baseline-no-defense.csv'

file1_column_name = 'baseline'
file2_column_name = 'baseline no attack'
file4_column_name = 'baseline no defense'


random_class = False
column_name = 'best validation accuracy'

title = 'Baseline Best Validation Accuracy'
ylabel = 'Best Validation Accuracy'

# %%

df1 = pd.read_csv(filepath1)
df2 = pd.read_csv(filepath2)
df4 = pd.read_csv(filepath4)


# Select the required columns from df1 and df2
df1_selected = df1[['epoch', column_name]].rename(columns={column_name: file1_column_name})
df2_selected = df2[['epoch', column_name]].rename(columns={column_name: file2_column_name})
df4_selected = df4[['epoch', column_name]].rename(columns={column_name: file4_column_name})


# Merge the DataFrames on the 'Epoch' column
df3 = pd.merge(pd.merge(df1_selected, df2_selected, on='epoch'), df4_selected, on='epoch')

# %%
import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(df3['epoch'], df3[file1_column_name], label=file1_column_name, color='black')  # Plot for df1
plt.plot(df3['epoch'], df3[file2_column_name], label=file2_column_name, color='red')  # Plot for df2
plt.plot(df3['epoch'], df3[file4_column_name], label=file4_column_name, color='blue')  # Plot for df2

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
