import torch
import numpy as np
import polars as pl
import torch.nn.functional as F


import sys
sys.path.append(f"../masters_thesis")
from analysis.helpers import Mapper, find_all_participants, read_data_for_participant
from typing import Tuple


def sliding_window(x: torch.Tensor, window_size: int, step_size: int = 1) -> torch.Tensor:
    # Slide over data    
    assert window_size <= x.shape[0], "window size cannot be larger than the dataset!"
    return x.unfold(0, window_size, step_size)


def prepare_data(df: pl.DataFrame, window_size: int) -> Tuple[torch.Tensor]:
    """
    Prepare windowed data for single dataframe.
    """
    data = torch.tensor([], dtype=torch.float32)
    mapper = Mapper()
    # Find where new sentences start
    idxs = np.append(df.select(pl.arg_where(pl.col("NEW_SENTENCE") == 1)).to_numpy().ravel(), 
                     [df.shape[0]]) 

    # Drop NEW_SENTENCE columns
    df = df.drop("NEW_SENTENCE")
    # Split the data by sentences
    for i, j in zip(idxs, idxs[1:]):
        # Slide over one sentence to create windows: [a, b, c, d] -> [[a, b], [b, c], [c, d]]
        try: 
            curr_data = sliding_window(torch.Tensor(df[i:j].to_numpy()), window_size=window_size)
            # Add to other data
            data = torch.cat((data, curr_data))
        except AssertionError:
            continue
    # Transpose to -> [num windows, window_size, features]
    data = data.transpose(-2, -1)
    X = data[:, :, 0].view(-1, window_size)
    y = data[:, :, 1:]

    # remove data for first keystrokes of the sequences
    mask = torch.zeros(*y.shape)
    mask = mask.index_fill_(1, torch.tensor([0]), 1)
    mask[:, :, 0] = 0
    y -= mask * y

    # Convert keystrokes to range [0; 100]
    X.apply_(lambda x: mapper.get_mapped_code_from_code(x))
    # Return X, y: X.shape -> [num_windows, window_size, 1], y.shape -> [num_windows, window_size, 4]
    return X, y


def create_dataloader(path: str, window_size: int, batch_size: int, shuffle: bool = True, limit: int = 20000) -> torch.utils.data.DataLoader:
    feature_columns = ["NEW_SENTENCE", "KEYCODE", "HOLD_TIME", "PRESS_PRESS_TIME", "RELEASE_PRESS_TIME", "RELEASE_RELEASE_TIME"]
    max_time = 3500 # 3.5 seconds in ms
    min_time = -1000 # 1 seconds in ms
    df_size_lim = 20
    X = torch.tensor([], dtype=torch.float32)
    y = torch.tensor([], dtype=torch.float32)
    participants = find_all_participants(path)
    for i, participant in enumerate(participants):
        if i % 1000 == 0:
            print(f"Processed: {i} / {len(participants)}")
        try:
            df = read_data_for_participant(participant, path, drop_timestamps=True, columns_to_read=["TEST_SECTION_ID", "KEYCODE", "RELEASE_TIME", "PRESS_TIME"])[feature_columns]
            if df.shape[0] < df_size_lim:
                continue
            if (df.to_numpy().max() > max_time) or (df.to_numpy().min() < min_time):
                continue
            X_curr, y_curr = prepare_data(df, window_size)
        except TypeError:
            continue
        X, y = torch.cat((X, X_curr)), torch.cat((y, y_curr))
        if i == limit - 1:
            break
            
    min_y = y.min()
    max_y = y.max()
    y_norm = (y - min_y) / (max_y - min_y)  # normalize to [0; 1]

    dataset = torch.utils.data.TensorDataset(X.int(), y_norm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
