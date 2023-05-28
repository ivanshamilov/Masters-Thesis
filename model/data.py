import torch
import numpy as np
import polars as pl
import torch.nn.functional as F

from tqdm import tqdm

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


class Dataset():
    def __init__(self, path: str, window_size: int, batch_size: int, shuffle: bool = True, limit: int = 1000, norm: bool = True, train_size: float = 0.8):
        self.path = path
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.limit = limit
        self.norm = norm
        self.feature_columns = ["NEW_SENTENCE", "KEYCODE", "HOLD_TIME", "RELEASE_PRESS_TIME"]
        self.max_time = 3000
        self.min_time = -1500
        self.df_size_lim = 25
        self.participants = find_all_participants(self.path)
        self.train_size = train_size
        self.valid_size = (1 - self.train_size) / 2
        self.test_size =  (1 - self.train_size) / 2

    def data_for_participant(self, participant):
        return read_data_for_participant(participant, self.path, drop_timestamps=True, 
                                         columns_to_read=["TEST_SECTION_ID", "KEYCODE", "RELEASE_TIME", "PRESS_TIME"])[self.feature_columns]  
    
    def _norm_data(self, X, y):
        X /= max(Mapper().inner_mapping.values())
        y = y * 10 ** -3   # convert keystroke times to seconds
        min_y = y.min()
        max_y = y.max()
        print(f"Min y: {min_y}, Max y: {max_y}")
        y = (y - min_y) / (max_y - min_y)  # normalize to [0; 1]

        return X, y
    
    def _find_negatives(self, main_participant, size):
        negative_X = torch.tensor([], dtype=torch.float32)
        negative_Y = torch.tensor([], dtype=torch.float32)

        other_participants = self.participants[:]
        other_participants.remove(main_participant) 
        np.random.shuffle(other_participants)

        for other_participant in other_participants:
            other_participant = self.data_for_participant(other_participant)
            if not self._df_sanity(other_participant):
                continue
            X, Y = prepare_data(other_participant, self.window_size)
            negative_X, negative_Y = torch.cat((negative_X, X)), torch.cat((negative_Y, Y))
            if negative_X.shape[0] > size:
                break

        return negative_X[:size], negative_Y[:size]
    
    def _df_sanity(self, df):
        if (df.shape[0] < self.df_size_lim) or (df.to_numpy().max() > self.max_time) or (df.to_numpy().min() < self.min_time):
            return False
        return True

    def create_dataset(self):
        X = torch.tensor([], dtype=torch.float32)
        y = torch.tensor([], dtype=torch.float32)
        i = 0 

        for participant in self.participants:
            try:
                df = self.data_for_participant(participant)
                if not self._df_sanity(df):
                    continue
                X_curr, y_curr = prepare_data(df, self.window_size)
            except TypeError:
                continue
            X, y = torch.cat((X, X_curr)), torch.cat((y, y_curr))
            i += 1
            if i == self.limit:
                break
        
        if self.norm:
            X, y = self._norm_data(X, y)
        
        dataset = torch.utils.data.TensorDataset(X.int(), y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        print(f"Number of batches: {len(dataloader)}, Number of instances: {len(dataloader.dataset)}")
        return dataloader

    def create_triplet_dataset(self):
        X = torch.tensor([], dtype=torch.float32)
        y = torch.tensor([], dtype=torch.float32)
        data = torch.tensor([], dtype=torch.float32)
        i = 0
        for participant in self.participants:
            main_participant = self.data_for_participant(participant)
            if not self._df_sanity(main_participant):
                continue
            try: 
                anchor_X, anchor_Y = prepare_data(main_participant, self.window_size)
                shuffle_idx = torch.randperm(anchor_X.shape[0])
                positive_X, positive_Y = anchor_X[shuffle_idx].view(anchor_X.size()), anchor_Y[shuffle_idx].view(anchor_Y.size())
                negative_X, negative_Y = self._find_negatives(participant, positive_X.shape[0])
            except TypeError:
                continue
            curr_X, curr_Y = torch.cat((anchor_X, positive_X, negative_X), dim=1), torch.cat((anchor_Y, positive_Y, negative_Y), dim=1)
            X, y = torch.cat((X, curr_X)), torch.cat((y, curr_Y))
            i += 1
            if i == self.limit:
                break

        if self.norm:
            X, y = self._norm_data(X, y)

        dataset = torch.cat((X.view(*X.shape, 1), y), dim=-1)  # keycodes are also the input features for TypeNet

        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[self.train_size, self.valid_size, self.test_size])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)

        print(f"Number of batches in train loader: {len(train_dataloader)}, Number of instances in train loader: {len(train_dataloader.dataset)}")
        print(f"Number of batches in valid loader: {len(valid_dataloader)}, Number of instances in valid loader: {len(valid_dataloader.dataset)}")
        print(f"Number of batches in test loader: {len(test_dataloader)}, Number of instances in test loader: {len(test_dataloader.dataset)}")

        return train_dataloader, valid_dataloader, test_dataloader
