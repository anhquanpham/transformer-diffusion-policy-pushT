import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['train_loss1000_plot.xlsx', 'train_loss2000_plot.xlsx', 'train_loss3000_plot.xlsx']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_excel(file_name, header=None)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Define consistent colors for each seed
colors = ['blue', 'green', 'orange']

# --- Plot 1: All Losses on One Plot ---
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'Seed = {file_names[i][10:14]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Loss of all 3 seeds')
plt.legend()
plt.grid(True)
# plt.show()  # Remove or comment out plt.show()
plt.savefig('trans_all_losses.png')  # Save the plot as all_losses.png


# --- Plot 3: Separate Plots for Individual Losses ---
for i, df in enumerate(dfs):
    plt.figure(figsize=(10, 6))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'Seed = {file_names[i][10:14]}', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'TransEncoder - Seed = {file_names[i][10:14]}')
    plt.legend()
    plt.grid(True)
    # plt.show()  # Remove or comment out plt.show()
    plt.savefig(f'trans_seed_{file_names[i][10:14]}.png')  # Save with seed in filename


# --- Plot 4: Mean Loss with Shaded Range ---
plt.figure(figsize=(10, 6))
epoch_values = dfs[0].iloc[:, 0]
loss_values = [df.iloc[:, 1].values for df in dfs]
mean_loss = np.mean(loss_values, axis=0)
std_loss = np.std(loss_values, axis=0)

plt.plot(epoch_values, mean_loss, label='Mean Loss', color='purple')  # Example color
plt.fill_between(epoch_values, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3, color='red', label='Loss Range')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Mean Loss with Shaded Range')
plt.legend()
plt.grid(True)
# plt.show()  # Remove or comment out plt.show()
plt.savefig('trans_mean_loss_shaded.png')  # Save the plot

# Create a plot for the mean loss with shaded range for the last 10 epochs
plt.figure(figsize=(10, 6))

# Assuming all dataframes have the same number of epochs
num_epochs = dfs[0].shape[0]  # Get the total number of epochs
last_10_epochs = epoch_values[-11:]  # Get the last 10 epoch values

# Calculate mean and standard deviation for the last 10 epochs
mean_loss_last_10 = np.mean([df.iloc[-11:, 1].values for df in dfs], axis=0)
std_loss_last_10 = np.std([df.iloc[-11:, 1].values for df in dfs], axis=0)

# Plot the mean loss and shaded range for the last 10 epochs
plt.plot(last_10_epochs, mean_loss_last_10, label='Mean Loss (Last 10 Epochs)')
plt.fill_between(last_10_epochs,
                 mean_loss_last_10 - std_loss_last_10,
                 mean_loss_last_10 + std_loss_last_10,
                 alpha=0.3, label='Loss Range (Last 10 Epochs)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Mean Loss with Shaded Range (Last 10 Epochs)')
plt.legend()
plt.grid(True)
plt.savefig('trans_mean_loss_last_10_epochs.png')  # Save the plot as mean_loss_last_10_epochs.png

# Create a plot for the mean loss with shaded range for the first 11 epochs
plt.figure(figsize=(10, 6))

# Assuming all dataframes have at least 11 epochs
first_11_epochs = epoch_values[:11]  # Get the first 11 epoch values

# Calculate mean and standard deviation for the first 11 epochs
mean_loss_first_11 = np.mean([df.iloc[:11, 1].values for df in dfs], axis=0)
std_loss_first_11 = np.std([df.iloc[:11, 1].values for df in dfs], axis=0)

# Plot the mean loss and shaded range for the first 11 epochs
plt.plot(first_11_epochs, mean_loss_first_11, label='Mean Loss (First 10 Epochs)')
plt.fill_between(first_11_epochs,
                 mean_loss_first_11 - std_loss_first_11,
                 mean_loss_first_11 + std_loss_first_11,
                 alpha=0.3, color='red', label='Loss Range (First 10 Epochs)')  # Shaded region in red

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Mean Loss with Shaded Range (First 10 Epochs)')
plt.legend()
plt.grid(True)
plt.savefig('trans_mean_loss_first_10_epochs.png')  # Save the plot as mean_loss_first_11_epochs.png

# Assuming all dataframes have the same number of epochs
num_epochs = dfs[0].shape[0]  # Get the total number of epochs

# Check if there are at least 11 epochs
if num_epochs < 11:
    print("Not enough epochs to plot the last 11. Exiting.")
    exit()

plt.figure(figsize=(10, 6))

# Get the last 11 epoch values
last_11_epochs = epoch_values[-11:]

# Calculate the min and max loss values for the last 11 epochs for each seed
loss_values_min_last_11 = np.min([df.iloc[-11:, 1].values for df in dfs], axis=0)
loss_values_max_last_11 = np.max([df.iloc[-11:, 1].values for df in dfs], axis=0)

# Fill the shaded region for the last 11 epochs
plt.fill_between(last_11_epochs, loss_values_min_last_11, loss_values_max_last_11, alpha=0.3, color='red', label='Loss Range (Last 11 Epochs)')

# Plot each seed's loss for the last 11 epochs
for i, df in enumerate(dfs):
    plt.plot(last_11_epochs, df.iloc[-11:, 1], label=f'Seed = {file_names[i][10:14]}', color=colors[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Shaded Plot of Losses for Last 10 Epochs')
plt.legend()
plt.grid(True)
plt.savefig('trans_shaded_plot_last_10_epochs.png')  # Save the plot as mean_loss_first_10_epochs.png


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_names = ['train_loss1000_plot.xlsx', 'train_loss2000_plot.xlsx', 'train_loss3000_plot.xlsx']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_excel(file_name, header=None)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Define consistent colors for each seed
colors = ['blue', 'green', 'orange']


# --- Plot for the first 11 epochs with shaded region ---
plt.figure(figsize=(10, 6))

# Assuming all dataframes have at least 11 epochs
first_11_epochs = dfs[0].iloc[:11, 0]  # Get the first 11 epoch values

# Calculate the min and max loss values for the first 11 epochs for each seed
loss_values_min_first_11 = np.min([df.iloc[:11, 1].values for df in dfs], axis=0)
loss_values_max_first_11 = np.max([df.iloc[:11, 1].values for df in dfs], axis=0)

# Fill the shaded region for the first 11 epochs
plt.fill_between(first_11_epochs, loss_values_min_first_11, loss_values_max_first_11, alpha=0.3, color='red', label='Loss Range (First 10 Epochs)')

# Plot each seed's loss for the first 11 epochs
for i, df in enumerate(dfs):
    plt.plot(first_11_epochs, df.iloc[:11, 1], label=f'Seed = {file_names[i][10:14]}', color=colors[i])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TransEncoder - Shaded Plot of Losses for First 10 Epochs')
plt.legend()
plt.grid(True)
plt.savefig('trans_shaded_plot_first_10_epochs.png')  # Save the plot as mean_loss_first_10_epochs.png

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['training_loss_seed_1000.xlsx', 'training_loss_seed_2000.xlsx', 'training_loss_seed_3000.xlsx']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_excel(file_name, header=None)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit()  # Exit if a file is not found

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

colors = ['blue', 'green', 'orange']

# --- Plot 1: All Losses on One Plot ---
plt.figure(figsize=(10, 6))
for i, df in enumerate(dfs):
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'Seed = {file_names[i][-9:-5]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Original - Loss of all 3 seeds')
plt.legend()
plt.grid(True)
plt.savefig('original_all_losses.png')

# --- Plot 3: Separate Plots for Individual Losses ---
for i, df in enumerate(dfs):
    plt.figure(figsize=(10, 6))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'Seed = {file_names[i][-9:-5]}', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Original - Seed = {file_names[i][-9:-5]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'original_seed_{file_names[i][-9:-5]}.png')

# --- Plot 4: Mean Loss with Shaded Range ---
plt.figure(figsize=(10, 6))
epoch_values = dfs[0].iloc[:, 0]
loss_values = [df.iloc[:, 1].values for df in dfs]
mean_loss = np.mean(loss_values, axis=0)
std_loss = np.std(loss_values, axis=0)

plt.plot(epoch_values, mean_loss, label='Mean Loss', color='purple')
plt.fill_between(epoch_values, mean_loss - std_loss, mean_loss + std_loss, alpha=0.3, color='red', label='Loss Range')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Original - Mean Loss with Shaded Range')
plt.legend()
plt.grid(True)
plt.savefig('original_mean_loss_shaded.png')

#Last 10
plt.figure(figsize=(10, 6))
last_10_epochs = epoch_values[-11:]
mean_loss_last_10 = np.mean([df.iloc[-11:, 1].values for df in dfs], axis=0)
std_loss_last_10 = np.std([df.iloc[-11:, 1].values for df in dfs], axis=0)
plt.plot(last_10_epochs, mean_loss_last_10, label='Mean Loss (Last 10 Epochs)')
plt.fill_between(last_10_epochs, mean_loss_last_10 - std_loss_last_10, mean_loss_last_10 + std_loss_last_10, alpha=0.3, label='Loss Range (Last 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Original - Mean Loss with Shaded Range (Last 10 Epochs)')
plt.legend()
plt.grid(True)
plt.savefig('original_mean_loss_last_10_epochs.png')

#First 10
plt.figure(figsize=(10, 6))
first_10_epochs = epoch_values[:11]
mean_loss_first_10 = np.mean([df.iloc[:11, 1].values for df in dfs], axis=0)
std_loss_first_10 = np.std([df.iloc[:11, 1].values for df in dfs], axis=0)
plt.plot(first_10_epochs, mean_loss_first_10, label='Mean Loss (First 10 Epochs)')
plt.fill_between(first_10_epochs, mean_loss_first_10 - std_loss_first_10, mean_loss_first_10 + std_loss_first_10, alpha=0.3, label='Loss Range (First 10 Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Original - Mean Loss with Shaded Range (First 10 Epochs)')
plt.legend()
plt.grid(True)
plt.savefig('original_mean_loss_first_10_epochs.png')

def plot_first_last_epochs(dfs, file_names, epoch_slice, title, filename):
    plt.figure(figsize=(10, 6))
    epoch_values = dfs[0].iloc[epoch_slice, 0]
    loss_values_min = np.min([df.iloc[epoch_slice, 1].values for df in dfs], axis=0)
    loss_values_max = np.max([df.iloc[epoch_slice, 1].values for df in dfs], axis=0)
    plt.fill_between(epoch_values, loss_values_min, loss_values_max, alpha=0.3, color='red', label='Loss Range')
    colors = ['blue', 'green', 'orange']
    for i, df in enumerate(dfs):
        plt.plot(epoch_values, df.iloc[epoch_slice, 1], label=f'Seed = {file_names[i][-9:-5]}', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)

#Plot first 10
plot_first_last_epochs(dfs, file_names, slice(0,11), 'Original - Shaded Plot of Losses for First 10 Epochs', 'original_shaded_plot_first_10_epochs.png')
#Plot last 10
plot_first_last_epochs(dfs, file_names, slice(-11,None), 'Original - Shaded Plot of Losses for Last 10 Epochs', 'original_shaded_plot_last_10_epochs.png')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['2trans_scores_and_steps_1000.csv', '2trans_scores_and_steps_2000.csv', '2trans_scores_and_steps_3000.csv']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_csv(file_name)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit()

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Combine all dataframes into one
combined_df = pd.concat(dfs)

# Option 1: Using pd.set_option to display all rows
# pd.set_option("display.max_rows", None)
#print(combined_df)
# pd.reset_option("display.max_rows")  # Reset to default

# Define score ranges (updated)
score_bins = np.arange(0, 1.05, 0.05)

# --- Plot for combined data ---
# Calculate average score
# print(combined_df['Score'].shape)
avg_score = combined_df['Score'].mean()
# Count the number of times score = 1
num_score_1 = combined_df['Score'][combined_df['Score'] == 1].count()
plt.figure(figsize=(10, 6))
plt.hist(combined_df['Score'], bins=score_bins, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'TransEncoder - Combined Score Distribution (Avg: {avg_score:.2f}, Score=1 count: {num_score_1})')
plt.xticks(score_bins)
plt.grid(True)
plt.savefig('2trans_score_distribution_combined.png')
plt.show()

# --- Plot for individual files ---
for i, df in enumerate(dfs):
    # Calculate average score for individual file
    # print(df['Score'])  # Print the 'Score' column of each individual df
    # print(df['Score'].shape)
    avg_score_individual = df['Score'].mean()
    # Count the number of times score = 1 for individual file
    num_score_1_individual = df['Score'][df['Score'] == 1].count()
    plt.figure(figsize=(10, 6))
    plt.hist(df['Score'], bins=score_bins, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'TransEncoder - Score Distribution - Seed = {file_names[i][24:-4]} (Avg: {avg_score_individual:.2f}, Score=1 count: {num_score_1_individual})')
    plt.xticks(score_bins)
    plt.grid(True)
    plt.savefig(f'2trans_score_distribution_{file_names[i][24:-4]}.png')  # Save with filename
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['2trans_scores_and_steps_1000.csv', '2trans_scores_and_steps_2000.csv', '2trans_scores_and_steps_3000.csv']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_csv(file_name)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit()

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Combine all dataframes into one
combined_df = pd.concat(dfs)

# Define step ranges
step_bins = [0, 100, 150, 200, float('inf')]
step_labels = ['Under 100', '100-150', '150-200', '201+']  # Changed label for clarity

# --- Pie chart for combined data ---
mean_steps_combined = combined_df['Steps'].mean()
combined_df['Step Category'] = pd.cut(combined_df['Steps'], bins=step_bins, labels=step_labels, right=False)
step_counts_combined = combined_df['Step Category'].value_counts()

# Create labels with counts
labels_with_counts = [f'{label} ({count})' for label, count in zip(step_counts_combined.index, step_counts_combined.values)]

plt.figure(figsize=(8, 8))
plt.pie(step_counts_combined, labels=labels_with_counts, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title(f'TransEncoder - Combined Step Distribution (Mean Steps: {mean_steps_combined:.2f})')
plt.legend(labels_with_counts, loc="upper left", bbox_to_anchor=(1, 0.5))
plt.tight_layout() # Adjust layout to include legend
plt.savefig('2trans_step_distribution_pie_chart_combined.png')  # Save combined plot
plt.show()

# --- Pie charts for individual files ---
for i, df in enumerate(dfs):
    mean_steps_individual = df['Steps'].mean()
    df['Step Category'] = pd.cut(df['Steps'], bins=step_bins, labels=step_labels, right=False)
    step_counts_individual = df['Step Category'].value_counts()

    # Create labels with counts for individual files
    labels_with_counts_individual = [f'{label} ({count})' for label, count in zip(step_counts_individual.index, step_counts_individual.values)]

    plt.figure(figsize=(8, 8))
    plt.pie(step_counts_individual, labels=labels_with_counts_individual, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title(f'TransEncoder Step Distribution - Seed = {file_names[i][24:-4]} (Mean Steps: {mean_steps_individual:.2f})')
    plt.legend(labels_with_counts_individual, loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()  # Adjust layout to include legend
    plt.savefig(f'2trans_step_distribution_pie_chart_{file_names[i][15:-4]}.png')  # Save individual plots
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['original_scores_and_steps_1000.csv', 'original_scores_and_steps_2000.csv', 'original_scores_and_steps_3000.csv']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_csv(file_name)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit()

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Combine all dataframes into one
combined_df = pd.concat(dfs)

# Option 1: Using pd.set_option to display all rows
# pd.set_option("display.max_rows", None)
#print(combined_df)
# pd.reset_option("display.max_rows")  # Reset to default

# Define score ranges (updated)
score_bins = np.arange(0, 1.05, 0.05)

# --- Plot for combined data ---
# Calculate average score
# print(combined_df['Score'].shape)
avg_score = combined_df['Score'].mean()
# Count the number of times score = 1
num_score_1 = combined_df['Score'][combined_df['Score'] == 1].count()
plt.figure(figsize=(10, 6))
plt.hist(combined_df['Score'], bins=score_bins, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title(f'Original - Combined Score Distribution (Avg: {avg_score:.2f}, Score=1 count: {num_score_1})')
plt.xticks(score_bins)
plt.grid(True)
plt.savefig('original_score_distribution_combined.png')
plt.show()

# --- Plot for individual files ---
for i, df in enumerate(dfs):
    # Calculate average score for individual file
    # print(df['Score'])  # Print the 'Score' column of each individual df
    # print(df['Score'].shape)
    avg_score_individual = df['Score'].mean()
    # Count the number of times score = 1 for individual file
    num_score_1_individual = df['Score'][df['Score'] == 1].count()
    plt.figure(figsize=(10, 6))
    plt.hist(df['Score'], bins=score_bins, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Original - Score Distribution - Seed = {file_names[i][26:-4]} (Avg: {avg_score_individual:.2f}, Score=1 count: {num_score_1_individual})')
    plt.xticks(score_bins)
    plt.grid(True)
    plt.savefig(f'original_score_distribution_{file_names[i][26:-4]}.png')  # Save with filename
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_names = ['original_scores_and_steps_1000.csv', 'original_scores_and_steps_2000.csv', 'original_scores_and_steps_3000.csv']
dfs = []

for file_name in file_names:
    try:
        df = pd.read_csv(file_name)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit()

if len(dfs) != 3:
    print("Not all files were loaded successfully. Exiting.")
    exit()

# Combine all dataframes into one
combined_df = pd.concat(dfs)

# Define step ranges
step_bins = [0, 100, 150, 200, float('inf')]
step_labels = ['Under 100', '100-150', '150-200', '201+']  # Changed label for clarity

# --- Pie chart for combined data ---
mean_steps_combined = combined_df['Steps'].mean()
combined_df['Step Category'] = pd.cut(combined_df['Steps'], bins=step_bins, labels=step_labels, right=False)
step_counts_combined = combined_df['Step Category'].value_counts()

# Create labels with counts
labels_with_counts = [f'{label} ({count})' for label, count in zip(step_counts_combined.index, step_counts_combined.values)]

plt.figure(figsize=(8, 8))
plt.pie(step_counts_combined, labels=labels_with_counts, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title(f'Original - Combined Step Distribution (Mean Steps: {mean_steps_combined:.2f})')
plt.legend(labels_with_counts, loc="upper left", bbox_to_anchor=(1, 0.5))
plt.tight_layout() # Adjust layout to include legend
plt.savefig('original_step_distribution_pie_chart_combined.png')  # Save combined plot
plt.show()

# --- Pie charts for individual files ---
for i, df in enumerate(dfs):
    mean_steps_individual = df['Steps'].mean()
    df['Step Category'] = pd.cut(df['Steps'], bins=step_bins, labels=step_labels, right=False)
    step_counts_individual = df['Step Category'].value_counts()

    # Create labels with counts for individual files
    labels_with_counts_individual = [f'{label} ({count})' for label, count in zip(step_counts_individual.index, step_counts_individual.values)]

    plt.figure(figsize=(8, 8))
    plt.pie(step_counts_individual, labels=labels_with_counts_individual, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title(f'Original Step Distribution - Seed = {file_names[i][26:-4]} (Mean Steps: {mean_steps_individual:.2f})')
    plt.legend(labels_with_counts_individual, loc="upper left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()  # Adjust layout to include legend
    plt.savefig(f'original_step_distribution_pie_chart_{file_names[i][26:-4]}.png')  # Save individual plots
    plt.show()