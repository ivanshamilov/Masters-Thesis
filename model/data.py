import torch
import numpy as np
import polars as pl

WINDOW_SIZE = 16


def sliding_window(x: torch.Tensor, window_size: int, step_size: int = 1):
    # Slide over data    
    return x.unfold(0, window_size, step_size)


def prepare_data(df: pl.DataFrame, window_size: int = WINDOW_SIZE):
    """
    Prepare windowed data for single dataframe.
    """
    data = torch.tensor([], dtype=torch.float32)

    # Find where new sentences start
    idxs = np.append(df.select(pl.arg_where(pl.col("NEW_SENTENCE") == 1)).to_numpy().ravel(), 
                     [df.shape[0]]) 

    # Drop NEW_SENTENCE columns
    df = df.drop("NEW_SENTENCE")

    # Split the data by sentences
    for i, j in zip(idxs, idxs[1:]):
        # Slide over one sentence to create windows: [a, b, c, d] -> [[a, b], [b, c], [c, d]]
        curr_data = sliding_window(torch.Tensor(df[i:j].to_numpy()), window_size=window_size)
        # Add to other data
        data = torch.cat((data, curr_data))
    
    # Transpose to -> [num windows, window_size, features]
    data = data.transpose(-2, -1)

    # Return X, y: X.shape -> [num_windows, window_size, 4], y.shape -> [num_windows, window_size 1]
    return data[:, :, 1:], data[:, :, 0].view(-1, window_size, 1)