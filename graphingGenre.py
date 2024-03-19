import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# This gets the epoch_size, batch_size and the learning_rate from the filename
def extract_parameters_from_filename(filename):
    parts = filename.split('_')
    params = parts[0].split('-')
    epoch_size, batch_size, learning_rate = params[:3]
    return epoch_size, batch_size, learning_rate

def plot_metrics(folder_path):
    # These are the metrics for we are using for the genre model
    metrics = ['subset_accuracy', 'hamming_loss', 'f1_micro']
    data = []
    files = [f for f in os.listdir(folder_path) if f.endswith('_results.csv')]

    # Go through all the files and get the data we need
    for file in files:
        epoch_size, batch_size, learning_rate = extract_parameters_from_filename(file)
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        avg_metrics = df[['subset_accuracy', 'hamming_loss', 'f1_micro']].mean().to_dict()
        avg_metrics.update({
            "epoch_size": epoch_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })
        data.append(avg_metrics)
    
    # Put it into a dataframe
    data = pd.DataFrame(data)

    # Get the respective "best" values for each metric and print
    best_row_hamming = data.loc[data["hamming_loss"].idxmin()]
    best_row_f1 = data.loc[data["f1_micro"].idxmax()]
    best_row_subset = data.loc[data["subset_accuracy"].idxmax()]

    print(f"Best combinations:")
    print(best_row_subset[['epoch_size', 'batch_size', 'learning_rate', "subset_accuracy"]])
    print(best_row_hamming[['epoch_size', 'batch_size', 'learning_rate', "hamming_loss"]])
    print(best_row_f1[['epoch_size', 'batch_size', 'learning_rate', "f1_micro"]])

    # Produce the heatmap for each metric for each epoch size
    for metric in metrics:
        for epoch_size in sorted(data['epoch_size'].unique(), key=int):
            # Make a figure for each learning rate and title
            fig, axes = plt.subplots(1, len(data['learning_rate'].unique()), figsize=(12, 6))
            fig.suptitle(f'Heatmap of {metric} for Epoch Size {epoch_size}', fontsize=16)

            for i, learning_rate in enumerate(sorted(data['learning_rate'].unique(), key=float)):
                # Filter the data by the epoch and learning rate for plotting
                filtered_data = data[(data['epoch_size'] == epoch_size) & (data['learning_rate'] == learning_rate)]
                # Make sure we have the right selection process for each metric
                # Lowest value for hamming_loss, highest for everything else
                # Pivot_table aggregeates the table by unique values
                if metric == "hamming_loss":
                    pivot_df = filtered_data.pivot_table(index='batch_size', columns='learning_rate', values=metric, aggfunc=np.min)                
                else :
                    pivot_df = filtered_data.pivot_table(index='batch_size', columns='learning_rate', values=metric, aggfunc=np.max)
                # Plot the heatmap using the pivot and set the titles
                ax = sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".3f", ax = axes[i], cbar=i == len(data['learning_rate'].unique())-1)
                ax.set_title(f'LR: {learning_rate}')
                ax.set_xlabel('')
                ax.set_xticks([])
                # remove the y labels for everything but the first row for aestetic reasons
                if i != 0:
                    ax.set_ylabel('')
                    ax.set_yticks([])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.subplots_adjust(wspace=0)
            plt.show()

folder_path = 'genre_results'
plot_metrics(folder_path)