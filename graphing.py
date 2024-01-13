import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_parameters_from_filename(filename):
    parts = filename.split('_')
    params = parts[0].split('-')
    epoch_size, batch_size, learning_rate = params[:3]
    return epoch_size, batch_size, learning_rate

def plot_loss(folder_path, loss_threshold):
    files = [f for f in os.listdir(folder_path) if f.endswith('_results.csv')]
    data = []

    for file in files:
        epoch_size, batch_size, learning_rate = extract_parameters_from_filename(file)
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        if 'train' in file:
            loss_key = 'train_loss'
            loss_type = 'train'
        elif 'val' in file:
            loss_key = 'val_loss'
            loss_type = 'val'
        else:
            continue

        df = df[df[loss_key] <= loss_threshold]

        for epoch, loss in df[loss_key].items():
            data.append({
                "epoch_size": epoch_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "loss": loss,
                "loss_type": loss_type
            })

    data = pd.DataFrame(data)
    g = sns.FacetGrid(data, col='learning_rate', hue='batch_size',col_wrap=5)
    g.map(plt.plot, "epoch", "loss")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_ylim(0, 10)
    plt.show()

    grouped_df = data.groupby(['epoch_size', 'batch_size', 'learning_rate', 'loss_type']).agg({'loss': 'mean'}).reset_index()
    train_data = grouped_df[grouped_df['loss_type'] == 'train']
    val_data = grouped_df[grouped_df['loss_type'] == 'val']
    epoch_sizes = grouped_df['epoch_size'].unique()


    num_epoch_sizes = len(epoch_sizes)
    plt.figure(figsize=(15, num_epoch_sizes * 5))
    for idx, epoch_size in enumerate(epoch_sizes):
        train_heatmap = train_data[train_data['epoch_size'] == epoch_size].pivot(index='batch_size', columns='learning_rate', values='loss')
        val_heatmap = val_data[val_data['epoch_size'] == epoch_size].pivot(index='batch_size', columns='learning_rate', values='loss')

        plt.subplot(num_epoch_sizes, 2, idx * 2 + 1)
        sns.heatmap(train_heatmap, annot=True, fmt=".2f")
        plt.title(f"Training Loss Heatmap (Epoch Size {epoch_size})")

        plt.subplot(num_epoch_sizes, 2, idx * 2 + 2)
        sns.heatmap(val_heatmap, annot=True, fmt=".2f")
        plt.title(f"Validation Loss Heatmap (Epoch Size {epoch_size})")
    
    plt.tight_layout()
    plt.show()


    best_params = data[data['loss_type'] == 'val'].groupby(['epoch_size', 'batch_size', 'learning_rate']).agg({'loss': 'min'}).reset_index()
    best_combination = best_params.loc[best_params['loss'].idxmin()]
    print("Best Parameter Combination:")
    print(best_combination)

folder_path = 'results'
plot_loss(folder_path, 500)